"""
Background job queue worker for video processing.

This module manages asynchronous job processing using Redis and RQ,
handling video analysis, editing, and rendering tasks in the background.
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID, uuid4
from datetime import datetime

import redis
from rq import Queue, Worker
from rq.job import Job

from app.models.schemas import (
    ProcessingJob, ProcessingStatus, TemplateType, QualityPreset,
    VideoAnalysisResult, VideoTimeline, EditingTemplate, VideoFormat
)
from app.config.settings import get_settings
from app.services.multi_video_manager import MultiVideoProjectManager
# Remove top-level imports of heavy dependencies - import them only when needed
# from app.analyzer.engine import get_analysis_engine
# from app.timeline.builder import get_timeline_builder
# from app.editor.renderer import get_renderer
# from app.ingestion.storage import get_storage_client
# from app.templates.manager import get_template_manager
from pydantic import AnyUrl, HttpUrl, ValidationError
import traceback
import requests

# Configure MoviePy to use system FFMPEG
import os
os.environ['FFMPEG_BINARY'] = '/opt/homebrew/bin/ffmpeg'

from moviepy import VideoFileClip
import subprocess, sys

logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUID objects."""
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle pydantic HttpUrl and similar types
        if type(obj).__name__ == 'HttpUrl':
            return str(obj)
        # Generic fallback for any object with __str__ if it's a URL
        if obj.__class__.__name__.lower().endswith('url'):
            return str(obj)
        return super().default(obj)

class JobQueue:
    """
    Background job queue for video processing tasks.
    
    Manages asynchronous processing of video analysis, editing,
    and rendering tasks using Redis and RQ.
    """
    
    def __init__(self, redis_url: str, queue_name: str = "video_editing"):
        """
        Initialize the job queue.
        
        Args:
            redis_url: Redis connection URL
            queue_name: Name of the job queue
        """
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.settings = get_settings()
        
        # Initialize Redis connection
        self.redis_conn = redis.from_url(redis_url)
        
        # Test Redis connection
        try:
            if self.redis_conn is not None:
                self.redis_conn.ping()
                logger.info(f"Redis connection established successfully to {redis_url}")
            else:
                logger.error("Redis connection is not initialized after from_url.")
                raise Exception("Redis connection is None after from_url.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {redis_url}: {e}")
            raise
        
        # Initialize RQ queue
        self.queue = Queue(queue_name, connection=self.redis_conn)
        
        # Job storage in Redis
        self.jobs: Dict[UUID, ProcessingJob] = {}  # Keep for backward compatibility
        self.job_prefix = "job:"
        self.video_job_prefix = "video_job:"
        
        # Start Redis monitoring in background
        self._start_redis_monitoring()
    
    def _start_redis_monitoring(self):
        """Start Redis monitoring in background to log Redis activity."""
        logger.info("✅ [REDIS MONITOR] Redis monitoring will log operations in existing methods")
        
        # Instead of complex pubsub monitoring, we'll add logging to existing Redis operations
        # This is simpler and more reliable 

    async def health_check(self) -> bool:
        """
        Perform health check for the job queue.
        
        Returns:
            bool: True if queue is healthy
        """
        try:
            if not self.redis_conn:
                logger.error("Redis connection is not initialized.")
                return False
            # Test Redis connection
            self.redis_conn.ping()
            return True
        except Exception as e:
            logger.error(f"Job queue health check failed: {e}")
            return False
    
    def _job_to_json(self, job: ProcessingJob) -> str:
        """Convert job to JSON string for Redis storage."""
        # Convert datetime objects to ISO format strings
        job_dict = job.model_dump()
        if job_dict.get('created_at'):
            job_dict['created_at'] = job_dict['created_at'].isoformat()
        if job_dict.get('started_at'):
            job_dict['started_at'] = job_dict['started_at'].isoformat()
        if job_dict.get('completed_at'):
            job_dict['completed_at'] = job_dict['completed_at'].isoformat()
        
        return json.dumps(job_dict, cls=UUIDEncoder)
    
    def _json_to_job(self, job_json: str) -> ProcessingJob:
        """Convert JSON string from Redis to ProcessingJob object."""
        job_dict = json.loads(job_json)
        
        # Convert ISO format strings back to datetime objects
        if job_dict.get('created_at'):
            job_dict['created_at'] = datetime.fromisoformat(job_dict['created_at'])
        if job_dict.get('started_at'):
            job_dict['started_at'] = datetime.fromisoformat(job_dict['started_at'])
        if job_dict.get('completed_at'):
            job_dict['completed_at'] = datetime.fromisoformat(job_dict['completed_at'])
        
        # Convert string UUIDs back to UUID objects
        if job_dict.get('job_id'):
            job_dict['job_id'] = UUID(job_dict['job_id'])
        if job_dict.get('video_id'):
            job_dict['video_id'] = UUID(job_dict['video_id'])
        
        # Handle empty dict in analysis_result (convert to None)
        if job_dict.get('analysis_result') == {}:
            job_dict['analysis_result'] = None
        
        return ProcessingJob(**job_dict)
    
    def _save_job_to_redis(self, job: ProcessingJob) -> bool:
        logger = logging.getLogger(__name__)
        try:
            if not self.redis_conn:
                logger.error("Redis connection is not initialized.")
                return False
            
            # Validate job schema before saving
            if not _validate_job_schema(job):
                logger.error(f"[VALIDATION] Job {job.job_id} failed schema validation, not saving to Redis")
                return False
            
            job_key = f"{self.job_prefix}{job.job_id}"
            video_job_key = f"{self.video_job_prefix}{job.video_id}"
            
            job_json = self._job_to_json(job)
            logger.info(f"[REDIS] Saving job_id={job.job_id} to Redis. analysis_result={'set' if job.analysis_result else 'None'}")
            
            # Log Redis operations
            logger.info(f"🔍 [REDIS MONITOR] SET {job_key}")
            logger.info(f"🔍 [REDIS MONITOR] SET {video_job_key}")
            logger.info(f"🔍 [REDIS MONITOR] SADD jobs {job.job_id}")
            
            # Save job by job_id
            self.redis_conn.set(job_key, job_json)
            
            # Save job by video_id for quick lookup
            self.redis_conn.set(video_job_key, job_json)
            
            # Add to job list for enumeration
            self.redis_conn.sadd("jobs", str(job.job_id))
            
            # Log job details
            logger.info(f"🔍 [REDIS MONITOR] Job {job.job_id}: status={job.status.value}, progress={job.progress}%")
            
            return True
        except Exception as e:
            logger.error(f"[REDIS] Failed to save job {job.job_id} to Redis: {e}")
            return False
    
    def _load_job_from_redis(self, job_id: UUID) -> Optional[ProcessingJob]:
        logger = logging.getLogger(__name__)
        try:
            if not self.redis_conn:
                logger.error("Redis connection is not initialized.")
                return None
            job_key = f"{self.job_prefix}{job_id}"
            
            # Log Redis operation
            logger.info(f"🔍 [REDIS MONITOR] GET {job_key}")
            
            job_json = self.redis_conn.get(job_key)
            
            if job_json:
                job = self._json_to_job(job_json.decode('utf-8'))
                logger.info(f"[REDIS] Loaded job_id={job_id} from Redis. analysis_result={'set' if job.analysis_result else 'None'}")
                
                # Log job details
                logger.info(f"🔍 [REDIS MONITOR] Job {job_id}: status={job.status.value}, progress={job.progress}%")
                
                return job
            return None
        except Exception as e:
            logger.error(f"[REDIS] Failed to load job {job_id} from Redis: {e}")
            return None
    
    def _load_job_by_video_id_from_redis(self, video_id: UUID) -> Optional[ProcessingJob]:
        """Load job from Redis by video_id."""
        try:
            if not self.redis_conn:
                logger.error("Redis connection is not initialized.")
                return None
            video_job_key = f"{self.video_job_prefix}{video_id}"
            job_json = self.redis_conn.get(video_job_key)
            
            if job_json:
                return self._json_to_job(job_json.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Failed to load job for video {video_id} from Redis: {e}")
            # Don't raise exception for status checks - just return None
            return None
    
    def _get_all_job_ids_from_redis(self) -> List[UUID]:
        """Get all job IDs from Redis."""
        try:
            if not self.redis_conn:
                logger.error("Redis connection is not initialized.")
                return []
            job_ids = set(self.redis_conn.smembers("jobs"))  # type: ignore
            return [UUID(job_id.decode('utf-8')) for job_id in job_ids]
        except Exception as e:
            logger.error(f"Failed to get job IDs from Redis: {e}")
            return []
    
    async def _update_project_status(self, project_id: str) -> None:
        """Update the project status to reflect current job progress."""
        try:
            from uuid import UUID
            project_uuid = UUID(project_id)
            multi_video_manager = MultiVideoProjectManager()
            await multi_video_manager.get_project_status(project_uuid)
            logger.info(f"[PROGRESS] Updated project {project_id} status")
        except Exception as e:
            logger.error(f"[PROGRESS] Failed to update project {project_id} status: {e}")
    
    async def create_analysis_job(
        self,
        video_id: UUID,
        template_type: Optional[TemplateType] = None,
        analysis_options: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None
    ) -> ProcessingJob:
        metadata = {
            "job_type": "analysis",
            "analysis_type": "gemini",  # Set to use Gemini analysis
            "analysis_options": analysis_options or {}
        }
        
        # Add project_id to metadata if provided (for multi-video projects)
        if project_id:
            metadata["project_id"] = project_id
            
        job = ProcessingJob(
            job_id=uuid4(),
            video_id=video_id,
            status=ProcessingStatus.PENDING,
            template_type=template_type or TemplateType.BEAT_MATCH,
            quality_preset=QualityPreset.HIGH,
            metadata=metadata
        )
        if self._save_job_to_redis(job):
            self.jobs[job.job_id] = job  # backward compatibility
            return job
        else:
            raise Exception("Failed to save job to Redis")

    async def create_editing_job(
        self,
        video_id: UUID,
        template_id: Optional[UUID] = None,
        template_type: TemplateType = TemplateType.BEAT_MATCH,
        custom_settings: Optional[Dict[str, Any]] = None,
        quality_preset: QualityPreset = QualityPreset.HIGH
    ) -> ProcessingJob:
        # Retrieve the completed analysis job for this video
        analysis_job = await self.get_job_by_video_id(video_id)
        if not analysis_job or not analysis_job.analysis_result:
            raise Exception("Analysis must be completed before editing.")
        job = ProcessingJob(
            job_id=uuid4(),
            video_id=video_id,
            status=ProcessingStatus.PENDING,
            template_type=template_type,
            quality_preset=quality_preset,
            analysis_result=analysis_job.analysis_result,  # Copy analysis result
            metadata={
                "job_type": "editing",
                "style": "tiktok",  # Default LLM style
                "edit_scale": 0.5,  # Default edit scale
                "custom_settings": custom_settings or {},
                "video_url": analysis_job.metadata.get("video_url")  # Copy video_url from analysis job
            }
        )
        if self._save_job_to_redis(job):
            self.jobs[job.job_id] = job  # backward compatibility
            return job
        else:
            raise Exception("Failed to save job to Redis")

    async def create_multi_video_editing_job(
        self,
        video_id: UUID,
        template_type: TemplateType = TemplateType.BEAT_MATCH,
        custom_settings: Optional[Dict[str, Any]] = None,
        quality_preset: QualityPreset = QualityPreset.HIGH,
        analysis_result: Optional[Any] = None
    ) -> ProcessingJob:
        """Create an editing job for multi-video projects without single video validation."""
        job = ProcessingJob(
            job_id=uuid4(),
            video_id=video_id,
            status=ProcessingStatus.PENDING,
            template_type=template_type,
            quality_preset=quality_preset,
            analysis_result=analysis_result,  # Use provided analysis result
            metadata={
                "job_type": "multi_video_editing",
                "style": "tiktok",  # Default LLM style
                "edit_scale": 0.5,  # Default edit scale
                "custom_settings": custom_settings or {},
                "multi_video_context": custom_settings.get("multi_video_context") if custom_settings else None,
            }
        )
        if self._save_job_to_redis(job):
            self.jobs[job.job_id] = job  # backward compatibility
            return job
        else:
            raise Exception("Failed to save job to Redis")

    def enqueue_job(self, job: ProcessingJob) -> bool:
        try:
            logger.info(f"[ENQUEUE] Attempting to enqueue job {job.job_id} of type {job.metadata.get('job_type')}")
            job.status = ProcessingStatus.PENDING
            job.started_at = datetime.utcnow()
            if job.metadata.get("job_type") == "analysis":
                logger.info("[ENQUEUE] Enqueuing analysis job in RQ...")
                try:
                    rq_job = self.queue.enqueue(
                        process_analysis_job_standalone,  # standalone function
                        str(job.job_id),
                        job_timeout=self.settings.job_timeout
                    )
                    logger.info(f"[ENQUEUE] RQ job created: {rq_job}")
                    # Store job metadata in RQ job for retrieval
                    rq_job.meta['job_data'] = job.model_dump()
                    rq_job.save_meta()
                except Exception as e:
                    logger.error(f"[ENQUEUE] Exception during RQ enqueue (analysis): {e}")
                    logger.error(f"[ENQUEUE] Failed to enqueue job {job.job_id}: {e}")
                    return False
            elif job.metadata.get("job_type") in ["editing", "multi_video_editing"]:
                logger.info(f"[ENQUEUE] Enqueuing {job.metadata.get('job_type')} job in RQ (standalone)...")
                try:
                    rq_job = self.queue.enqueue(
                        process_editing_job_standalone,  # standalone function (no subprocess)
                        str(job.job_id),
                        job_timeout=self.settings.job_timeout
                    )
                    logger.info(f"[ENQUEUE] RQ job created: {rq_job}")
                    # Store job metadata in RQ job for retrieval
                    rq_job.meta['job_data'] = job.model_dump()
                    rq_job.save_meta()
                except Exception as e:
                    logger.error(f"[ENQUEUE] Exception during RQ enqueue (editing): {e}")
                    logger.error(f"[ENQUEUE] Failed to enqueue job {job.job_id}: {e}")
                    return False
            elif job.metadata.get("job_type") == "cross_video_analysis":
                logger.info("[ENQUEUE] Enqueuing cross-video analysis job in RQ...")
                try:
                    rq_job = self.queue.enqueue(
                        process_cross_analysis_job_standalone,  # standalone function
                        str(job.job_id),
                        job_timeout=self.settings.job_timeout
                    )
                    logger.info(f"[ENQUEUE] RQ job created: {rq_job}")
                    # Store job metadata in RQ job for retrieval
                    rq_job.meta['job_data'] = job.model_dump()
                    rq_job.save_meta()
                except Exception as e:
                    logger.error(f"[ENQUEUE] Exception during RQ enqueue (cross_analysis): {e}")
                    logger.error(f"[ENQUEUE] Failed to enqueue job {job.job_id}: {e}")
                    return False
            else:
                logger.error(f"[ENQUEUE] Unknown job type: {job.metadata.get('job_type')}")
                return False
            # Restore custom Redis storage for job tracking
            self._save_job_to_redis(job)
            return True
        except Exception as e:
            logger.error(f"[ENQUEUE] Exception in enqueue_job: {e}")
            return False

    async def get_job_by_video_id(self, video_id: UUID) -> Optional[ProcessingJob]:
        return self._load_job_by_video_id_from_redis(video_id)

    async def list_jobs(
        self,
        page: int = 1,
        page_size: int = 10,
        status: Optional[ProcessingStatus] = None
    ) -> Tuple[List[ProcessingJob], int]:
        all_job_ids = self._get_all_job_ids_from_redis()
        jobs = []
        for job_id in all_job_ids:
            job = self._load_job_from_redis(job_id)
            if job:
                if status is None or job.status == status:
                    jobs.append(job)
        total_count = len(jobs)
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        return jobs[start:end], total_count
    
    async def _process_analysis_job(self, job_id: UUID):
        """Skip analysis process - new workflow uses create_robust_25_second_video.py directly."""
        logger.info(f"[SKIP ANALYSIS] Skipping analysis process for job_id={job_id} - using new workflow")
        
        try:
            # Load job from Redis
            job = self._load_job_from_redis(job_id)
            if not job:
                logger.error(f"[ANALYSIS JOB ERROR] Job {job_id} not found in Redis")
                return
            logger.info(f"[REDIS] Loaded job_id={job_id} from Redis. analysis_result={'set' if job.analysis_result else 'None'}")
            
            # Update job status to completed immediately
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100
            job.error_message = "Analysis skipped - using new workflow with create_robust_25_second_video.py"
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 100%: Analysis skipped - new workflow")

            # Auto-trigger cross-analysis for multi-video projects
            try:
                logger.info(f"[AUTO CROSS-ANALYSIS] Analysis job {job_id} completed, checking if cross-analysis should be triggered...")
                
                # Check if this job is part of a multi-video project
                project_id = job.metadata.get("project_id") if job.metadata else None
                if project_id:
                    logger.info(f"[AUTO CROSS-ANALYSIS] Found project_id: {project_id}")
                    
                    # Get the multi-video project
                    from app.services.multi_video_manager import get_multi_video_manager
                    multi_video_manager = await get_multi_video_manager()
                    project = await multi_video_manager.get_project(UUID(project_id))
                    
                    if project:
                        # Skip analysis and cross-analysis phases - go directly to editing
                        logger.info(f"[AUTO WORKFLOW] Skipping analysis and cross-analysis phases, going directly to editing...")
                        
                        # Trigger editing job directly using create_robust_25_second_video.py
                        from app.api.routes import _trigger_editing_for_project
                        await _trigger_editing_for_project(project_id)
                        
                        logger.info(f"[AUTO WORKFLOW] Editing job triggered for project {project_id}")
                    else:
                        logger.warning(f"[AUTO CROSS-ANALYSIS] Project {project_id} not found")
                else:
                    logger.info(f"[AUTO CROSS-ANALYSIS] No project_id found in analysis job metadata - single video analysis")
                    
            except Exception as e:
                logger.error(f"[AUTO CROSS-ANALYSIS] Failed to check or trigger cross-analysis: {e}")

        except Exception as e:
            logger.error(f"[ANALYSIS JOB ERROR] Unexpected error: {e}")
            job = self._load_job_from_redis(job_id)
            if job:
                job.status = ProcessingStatus.FAILED
                job.error_message = f"Unexpected error: {e}"
                self._save_job_to_redis(job)
            return

    async def _simple_mock_analysis(self, video_path: str):
        """This method is deprecated. Use real analysis instead."""
        logger.error(f"[DEPRECATED] Mock analysis is no longer supported. Use real analysis for {video_path}")
        raise NotImplementedError("Mock analysis is deprecated. Please use real video analysis.")
    
    async def _download_video_for_analysis(self, job_id: UUID) -> str:
        """Download video for analysis."""
        try:
            from app.services import get_service_manager
            
            job = self._load_job_from_redis(job_id)
            if not job:
                raise Exception(f"Job {job_id} not found")
            
            # Get video file through service manager
            service_manager = get_service_manager()
            storage_client = await service_manager.get_storage()
            
            video_url = job.metadata.get("video_url") if job.metadata else None
            if not video_url:
                # Fallback to constructing the URL from video_id (for uploaded videos)
                video_url = f"uploads/{job.video_id}.mp4"
            
            temp_video_path = os.path.join(self.settings.temp_directory, f"{job.video_id}_source.mp4")
            success = await storage_client.download_file(video_url, temp_video_path)
            
            if not success:
                raise Exception("Failed to download video file")
            
            return temp_video_path
            
        except Exception as e:
            logger.error(f"Failed to download video for analysis: {e}")
            raise
    
    async def _process_editing_job(self, job_id: UUID) -> bool:
        """Process video editing job with enhanced error handling."""
        logger.info(f"[EDIT JOB START] _process_editing_job called with job_id={job_id}")
        
        try:
            # Load job from Redis
            job = self._load_job_from_redis(ensure_uuid(job_id))
            if not job:
                logger.error(f"[EDIT JOB ERROR] Job {job_id} not found in Redis")
                return False
            
            logger.info(f"[REDIS] Loaded job_id={job_id} from Redis. analysis_result={'set' if job.analysis_result else 'None'}")
            
            # Update job status to processing
            job.status = ProcessingStatus.EDITING
            job.progress = 20
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 20%: Status EDITING")

            # Check if we have an enhanced LLM plan and convert it to timeline first
            enhanced_llm_plan_json = job.metadata.get("enhanced_llm_plan_json")
            if enhanced_llm_plan_json and enhanced_llm_plan_json.strip():
                logger.info(f"✅ [ENHANCED LLM] Found enhanced LLM plan in job metadata, converting to timeline")
                try:
                    enhanced_plan_data = json.loads(enhanced_llm_plan_json)
                    
                    # Use the enhanced LLM editor to generate timeline
                    from app.editor.enhanced_llm_editor import create_enhanced_llm_editor
                    from app.models.schemas import EditStyle, VideoTimeline, TimelineSegment, EditingTemplate
                    
                    enhanced_editor = create_enhanced_llm_editor("openai")
                    style = EditStyle(enhanced_plan_data.get("style", "tiktok"))
                    
                    # Generate timeline from LLM plan
                    segments = []
                    for segment_data in enhanced_plan_data.get("segments", []):
                        segments.append(TimelineSegment(
                            start_time=segment_data.get("start_time", 0.0),
                            end_time=segment_data.get("end_time", 0.0),
                            source_video_id=UUID(segment_data.get("source_video_id", str(job.video_id))),
                            effects=segment_data.get("effects", []),
                            transition_in=segment_data.get("transition_in", "cross_dissolve"),
                            transition_out=segment_data.get("transition_out", "cross_dissolve"),
                        ))
                    
                    # Create a template for the timeline
                    template = EditingTemplate(
                        name=f"LLM Generated - {style.value}",
                        template_type=job.template_type,
                        description=f"LLM-generated template for {style.value} style editing"
                    )
                    
                    timeline = VideoTimeline(
                        video_id=job.video_id,
                        template=template,
                        segments=segments,
                        total_duration=enhanced_plan_data.get("target_duration", job.analysis_result.duration if job.analysis_result else 0.0)
                    )
                    
                    # Update job with new timeline
                    job.timeline = timeline
                    logger.info(f"✅ [ENHANCED LLM] Generated timeline from LLM plan with {len(segments)} segments")
                    
                except Exception as e:
                    logger.error(f"❌ [ENHANCED LLM] Failed to convert LLM plan to timeline: {e}")
                    logger.info(f"🔄 [ENHANCED LLM] Falling back to regular timeline processing")

            # Check if this is a multi-video project that needs fresh generation
            multi_video_context = job.metadata.get("multi_video_context")
            force_fresh_generation = multi_video_context is not None
            
            # Now check if we have a timeline (either from LLM plan or pre-existing)
            if job.timeline and job.timeline.segments and not force_fresh_generation:
                logger.info(f"✅ [ENHANCED TIMELINE] Using timeline with {len(job.timeline.segments)} segments")
                timeline = job.timeline
                
                # Log the timeline segments for debugging
                for i, segment in enumerate(timeline.segments):
                    logger.info(f"�� [ENHANCED TIMELINE] Segment {i+1}: {segment.start_time:.3f}s-{segment.end_time:.3f}s, Effects: {segment.effects}")
                
            else:
                # No timeline available - check if we should use new workflow
                use_new_workflow = job.metadata.get("custom_settings", {}).get("use_new_workflow", False)
                workflow_type = job.metadata.get("custom_settings", {}).get("workflow_type", "legacy")
                
                if use_new_workflow and workflow_type == "gemini_direct":
                    # Use new integrated multi-video editor
                    logger.info(f"🚀 [NEW WORKFLOW] Using integrated MultiVideoEditor for multi-video editing")
                    
                    # Get project information
                    project_id = job.metadata.get("custom_settings", {}).get("project_id")
                    video_ids = job.metadata.get("custom_settings", {}).get("video_ids", [])
                    
                    if not project_id or not video_ids:
                        raise Exception("Project ID or video IDs not available for new workflow")
                    
                    # Import the integrated multi-video editor
                    from app.editor.multi_video_editor import MultiVideoEditor
                    
                    # Convert video IDs to file paths (download from S3)
                    from app.services import get_service_manager
                    service_manager = get_service_manager()
                    storage = await service_manager.get_storage()
                    
                    video_paths = []
                    # Ensure temp directory exists
                    import os
                    os.makedirs("/tmp/video_editing", exist_ok=True)
                    
                    for video_id in video_ids:
                        # Download video from S3 to local temp file
                        temp_path = f"/tmp/video_editing/{video_id}_source.mp4"
                        try:
                            await storage.download_file(f"uploads/{video_id}.mp4", temp_path)
                            video_paths.append(temp_path)
                            logger.info(f"✅ [NEW WORKFLOW] Downloaded video {video_id} to {temp_path}")
                        except Exception as e:
                            logger.error(f"❌ [NEW WORKFLOW] Failed to download video {video_id}: {e}")
                            raise Exception(f"Failed to download video {video_id}: {e}")
                    
                    # Create multi-video editor and process videos
                    editor = MultiVideoEditor()
                    
                    # Update progress to 30% - starting video processing
                    job.progress = 30
                    job.status = ProcessingStatus.EDITING
                    self._save_job_to_redis(job)
                    logger.info(f"[PROGRESS] Set to 30%: Starting video processing")
                    
                    # Update project status to reflect job progress
                    await self._update_project_status(project_id)
                    
                    output_path = await editor.create_multi_video(video_paths, project_id)
                    
                    # Update progress to 80% - video processing completed
                    job.progress = 80
                    job.status = ProcessingStatus.EDITING
                    self._save_job_to_redis(job)
                    logger.info(f"[PROGRESS] Set to 80%: Video processing completed")
                    
                    # Update project status to reflect job progress
                    await self._update_project_status(project_id)
                    
                    if output_path and os.path.exists(output_path):
                        logger.info(f"✅ [NEW WORKFLOW] Video created successfully: {output_path}")
                        
                        # Upload the result to S3
                        output_filename = f"multi_video_output_{project_id}.mp4"
                        output_s3_key = f"outputs/{output_filename}"
                        
                        try:
                            await storage.upload_file(output_path, output_s3_key)
                            output_url = f"https://my-video-editing-app-2025.s3.amazonaws.com/{output_s3_key}"
                            logger.info(f"✅ [NEW WORKFLOW] Video uploaded to S3: {output_url}")
                            
                            # Update job with success
                            job.status = ProcessingStatus.COMPLETED
                            job.progress = 100
                            job.output_url = output_url
                            job.error_message = None
                            self._save_job_to_redis(job)
                            
                            # Update project status to reflect completion
                            await self._update_project_status(project_id)
                            
                            logger.info(f"✅ [NEW WORKFLOW] Multi-video editing completed successfully")
                            return
                            
                        except Exception as e:
                            logger.error(f"❌ [NEW WORKFLOW] Failed to upload video to S3: {e}")
                            raise Exception(f"Failed to upload video to S3: {e}")
                    else:
                        logger.error(f"❌ [NEW WORKFLOW] Video creation failed or output file not found")
                        raise Exception("Video creation failed")
                
                else:
                    # Use legacy workflow - generate a fresh one
                    logger.info(f"🔄 [TIMELINE] No timeline found, generating fresh LLM plan")
                    
                    # Extract style from metadata or use default
                    style = job.metadata.get("style", "tiktok")  # default to tiktok
                    
                    # Generate fresh LLM plan using enhanced LLM editor
                    from app.editor.enhanced_llm_editor import create_enhanced_llm_editor
                    from app.models.schemas import EditStyle
                
                if not job.analysis_result:
                    raise Exception("Analysis result not available")
                
                # Create enhanced LLM editor
                enhanced_editor = create_enhanced_llm_editor("openai")
                style_enum = EditStyle(style)
                
                # Get multi-video context if available
                multi_video_context = job.metadata.get("multi_video_context")
                
                # Generate enhanced editing plan
                enhanced_plan = enhanced_editor.generate_editing_plan(
                    analysis_result=job.analysis_result,
                    style=style_enum,
                    target_duration=job.metadata.get("target_duration"),
                    multi_video_context=multi_video_context
                )
                
                # Convert enhanced LLM plan to timeline
                from app.models.schemas import VideoTimeline, TimelineSegment, EditingTemplate
                segments = []
                
                # Handle enhanced plan format
                if hasattr(enhanced_plan, 'segments'):
                    # Enhanced plan has segments directly
                    for seg in enhanced_plan.segments:
                        segments.append(TimelineSegment(
                            start_time=seg.start_time,
                            end_time=seg.end_time,
                            source_video_id=seg.source_video_id,
                            effects=seg.effects,
                            transition_in=seg.transition_in,
                            transition_out=seg.transition_out,
                        ))
                else:
                    # Fallback to old format
                    for seg in enhanced_plan.get("segments", []):
                        segments.append(TimelineSegment(
                            start_time=seg.get("start_time", 0.0),
                            end_time=seg.get("end_time", 0.0),
                            source_video_id=UUID(seg.get("source_video_id", str(job.video_id))),
                            effects=seg.get("effects", []),
                            transition_in=seg.get("transition_in", "cross_dissolve"),
                            transition_out=seg.get("transition_out", "cross_dissolve"),
                        ))
                
                # Create a default template for the timeline
                default_template = EditingTemplate(
                    name=f"LLM Generated - {style}",
                    template_type=job.template_type,
                    description=f"LLM-generated template for {style} style editing"
                )
                
                timeline = VideoTimeline(
                    video_id=job.video_id,
                    template=default_template,
                    segments=segments,
                    total_duration=job.analysis_result.duration
                )
                
                # Update job with new timeline
                job.timeline = timeline
                logger.info(f"✅ [LLM EDIT] Generated fresh LLM timeline with {len(segments)} segments")
        
            job.progress = 50
            
            # Get video file
            from app.services import get_service_manager
            storage_client = await get_service_manager().get_storage()
            video_url = job.metadata.get("video_url")
            if not video_url:
                # Fallback to constructing the URL from video_id (for uploaded videos)
                video_url = f"uploads/{job.video_id}.mp4"
            
            settings = get_settings()
            temp_video_path = os.path.join(settings.temp_directory, f"{job.video_id}_source.mp4")
            
            # Handle download based on storage type
            if hasattr(storage_client, 'download_file') and callable(getattr(storage_client, 'download_file')):
                success = await storage_client.download_file(video_url, temp_video_path)
                if not success:
                    raise Exception("Failed to download video file")
            else:
                # Real storage - download the actual video file
                logger.error("Real video download not implemented. Please implement proper video storage.")
                raise NotImplementedError("Real video download not implemented. Please implement proper video storage.")
            
            job.progress = 50
            self._save_job_to_redis(job)
            
            # Render video
            logger.info(f"🎬 [EDIT JOB] Starting video rendering...")
            logger.info(f"📁 [EDIT JOB] Source video path: {temp_video_path}")
            logger.info(f"📁 [EDIT JOB] Timeline segments: {len(timeline.segments)}")
            logger.info(f"⚙️ [EDIT JOB] Quality preset: {job.quality_preset}")
            
            job.status = ProcessingStatus.RENDERING
            
            # Check if this is a multi-video workflow
            is_multi_video = job.metadata.get("job_type") == "multi_video_editing"
            
            # Use universal SimpleVideoRenderer for all workflows (single and multi-video)
            logger.info(f"🎬 [LLM EDIT] Using universal SimpleVideoRenderer for {'multi-video' if is_multi_video else 'single-video'} workflow")
            try:
                from app.editor.renderer_simple import SimpleVideoRenderer
                renderer = SimpleVideoRenderer()
            except Exception as e:
                logger.error(f"❌ [EDIT JOB] Failed to initialize universal SimpleVideoRenderer: {e}")
                raise Exception(f"Failed to initialize renderer: {e}")
            
            output_filename = f"{job.video_id}_edited.mp4"
            temp_output_path = os.path.join(settings.temp_directory, output_filename)
            logger.info(f"📁 [EDIT JOB] Output path: {temp_output_path}")
            
            logger.info(f"🎬 [EDIT JOB] Calling renderer.render_video...")
            
            # Create progress callback for renderer
            def render_progress_callback(progress: int, message: str):
                try:
                    if job:
                        # Map renderer progress (0-100%) to job progress (55-95%)
                        # This gives us 40% range for rendering (55-95%)
                        job_progress = 55 + int(progress * 0.4)
                        job.progress = job_progress
                        
                        # Save to Redis with error handling
                        try:
                            self._save_job_to_redis(job)
                            logger.info(f"🎬 [RENDER PROGRESS] Job progress updated to {job_progress}%: {message}")
                        except Exception as redis_error:
                            logger.error(f"❌ [RENDER PROGRESS] Failed to save progress to Redis: {redis_error}")
                    else:
                        logger.warning(f"🎬 [RENDER PROGRESS] No job object available for progress update")
                    
                    logger.info(f"🎬 [RENDER PROGRESS] {progress}%: {message}")
                except Exception as callback_error:
                    logger.error(f"❌ [RENDER PROGRESS] Progress callback error: {callback_error}")
            
            # Update progress to show rendering is starting
            job.progress = 55
            self._save_job_to_redis(job)
            logger.info(f"🎬 [EDIT JOB] Starting video rendering at 55% progress")
            logger.info(f"🎬 [EDIT JOB] Video path: {temp_video_path}")
            logger.info(f"🎬 [EDIT JOB] Output path: {temp_output_path}")
            logger.info(f"🎬 [EDIT JOB] Timeline segments: {len(timeline.segments)}")
            logger.info(f"🎬 [EDIT JOB] Quality preset: {job.quality_preset}")
            
            # Add detailed logging for multi-video detection
            if hasattr(renderer, '_is_multi_video_project'):
                is_multi = renderer._is_multi_video_project(timeline)
                logger.info(f"🎬 [EDIT JOB] Multi-video project detected: {is_multi}")
                if is_multi:
                    logger.info(f"🎬 [EDIT JOB] Will use multi-video rendering path")
                else:
                    logger.info(f"🎬 [EDIT JOB] Will use single-video rendering path")
            
            logger.info(f"🎬 [EDIT JOB] About to call renderer.render_video()...")
            success = await renderer.render_video(
                temp_video_path,
                timeline,
                temp_output_path,
                job.quality_preset,
                render_progress_callback
            )
            logger.info(f"🎬 [EDIT JOB] renderer.render_video returned: {success}")
            
            if not success:
                logger.error(f"❌ [EDIT JOB] Video rendering failed")
                raise Exception("Video rendering failed")
            
            logger.info(f"✅ [EDIT JOB] Video rendering completed successfully")
            
            job.progress = 80
            self._save_job_to_redis(job)
            
            # Upload rendered video
            output_key = f"processed/{job.video_id}_processed.mp4"
            output_url = await storage_client.upload_file(
                temp_output_path,
                output_key,
                content_type="video/mp4"
            )
            
            # Convert string URL to HttpUrl if needed
            from pydantic import AnyUrl
            if isinstance(output_url, str):
                job.output_url = AnyUrl(output_url)
            else:
                job.output_url = output_url
            job.progress = 100
            
            # Update status
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            self._save_job_to_redis(job)
            
            # For multi-video projects, update the project status
            if job.metadata.get("job_type") == "multi_video_editing":
                try:
                    from app.services.multi_video_manager import get_multi_video_manager
                    multi_video_manager = await get_multi_video_manager()
                    project_id = job.metadata.get("project_id")
                    
                    if project_id:
                        logger.info(f"🎬 [MULTI-VIDEO] Updating project {project_id} status after editing completion")
                        
                        # Update project status to completed
                        await multi_video_manager.update_project_status(
                            project_id=UUID(project_id),
                            status="completed",
                            output_video_id=job.video_id
                        )
                        
                        logger.info(f"✅ [MULTI-VIDEO] Project {project_id} status updated to completed")
                    else:
                        logger.warning(f"⚠️ [MULTI-VIDEO] No project_id found in job metadata")
                        
                except Exception as e:
                    logger.error(f"❌ [MULTI-VIDEO] Failed to update project status: {e}")
            
            # Cleanup
            for temp_file in [temp_video_path, temp_output_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.info(f"Editing job {job_id} completed successfully")
            return True
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"❌ [EDIT JOB FAIL] job_id={job_id}")
            logger.error(f"❌ [EDIT JOB FAIL] Exception type: {type(e).__name__}")
            logger.error(f"❌ [EDIT JOB FAIL] Exception message: {str(e)}")
            logger.error(f"❌ [EDIT JOB FAIL] Full traceback:")
            logger.error(tb)
            
            # Log additional context for debugging
            logger.error(f"❌ [EDIT JOB FAIL] Job context:")
            try:
                job = self._load_job_from_redis(ensure_uuid(job_id))
                if job:
                    logger.error(f"❌ [EDIT JOB FAIL] Job status: {job.status}")
                    logger.error(f"❌ [EDIT JOB FAIL] Job progress: {job.progress}")
                    logger.error(f"❌ [EDIT JOB FAIL] Job metadata: {job.metadata}")
                    logger.error(f"❌ [EDIT JOB FAIL] Has analysis result: {job.analysis_result is not None}")
                    logger.error(f"❌ [EDIT JOB FAIL] Has timeline: {job.timeline is not None}")
                else:
                    logger.error(f"❌ [EDIT JOB FAIL] Could not load job from Redis")
            except Exception as context_error:
                logger.error(f"❌ [EDIT JOB FAIL] Failed to get job context: {context_error}")
            
            # Update job status to failed
            try:
                job = self._load_job_from_redis(ensure_uuid(job_id))
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    self._save_job_to_redis(job)
            except Exception as save_error:
                logger.error(f"❌ [EDIT JOB FAIL] Failed to save failed status: {save_error}")
            
            return False

    async def _process_cross_analysis_job(self, job_id: UUID):
        """Process cross-video analysis job."""
        from uuid import UUID  # Ensure UUID is available in this scope
        logger.info(f"[CROSS ANALYSIS JOB START] _process_cross_analysis_job called with job_id={job_id}")
        
        try:
            # Load job from Redis
            job = self._load_job_from_redis(job_id)
            if not job:
                logger.error(f"[CROSS ANALYSIS JOB ERROR] Job {job_id} not found in Redis")
                return False
            
            logger.info(f"[REDIS] Loaded cross-analysis job_id={job_id} from Redis")
            
            # Update job status to processing
            job.status = ProcessingStatus.ANALYZING
            job.progress = 10
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 10%: Cross-analysis started")

            # Extract project information from job metadata
            project_id = job.metadata.get("project_id")
            video_ids = job.metadata.get("video_ids", [])
            analysis_job_ids = job.metadata.get("analysis_job_ids", [])
            
            logger.info(f"[CROSS ANALYSIS] Processing project {project_id} with {len(video_ids)} videos")
            
            # Dynamically load analysis results from job IDs
            analysis_results = []
            for analysis_job_id in analysis_job_ids:
                analysis_job = self._load_job_from_redis(UUID(analysis_job_id))
                if analysis_job and analysis_job.analysis_result:
                    analysis_results.append(analysis_job.analysis_result)
                    logger.info(f"[CROSS ANALYSIS] Loaded analysis result from job {analysis_job_id}")
                else:
                    logger.warning(f"[CROSS ANALYSIS] Analysis job {analysis_job_id} not found or has no result")
            
            logger.info(f"[CROSS ANALYSIS] Loaded {len(analysis_results)} analysis results")
            
            # Import the cross-video analysis function
            from app.api.routes import _perform_cross_video_analysis
            
            # Perform cross-video analysis
            job.progress = 50
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 50%: Cross-video analysis in progress")
            
            cross_analysis_result = await _perform_cross_video_analysis(
                project_id=project_id,
                video_ids=video_ids,
                analysis_results=analysis_results,
                settings=job.metadata.get("cross_analysis_settings", {})
            )
            
            # Store the cross-analysis result in metadata, not in analysis_result field
            # (analysis_result expects VideoAnalysisResult, not CrossVideoAnalysisResult)
            job.metadata["cross_analysis_result"] = cross_analysis_result.dict()
            job.analysis_result = None  # Clear analysis_result for cross-analysis jobs
            job.progress = 90
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 90%: Cross-analysis result stored in metadata")
            
            # Mark job as completed
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.utcnow()
            self._save_job_to_redis(job)
            logger.info(f"[PROGRESS] Set to 100%: Cross-analysis completed")
            
            logger.info(f"[CROSS ANALYSIS JOB SUCCESS] Cross-video analysis completed successfully for job {job_id}")
            
            # Check if all individual analysis jobs are completed before triggering LLM recommendation
            try:
                logger.info(f"[AUTO LLM TRIGGER] Cross-analysis completed, checking if all analysis jobs are ready...")
                
                # Get the project ID from the cross-analysis job metadata
                project_id = job.metadata.get('project_id')
                if project_id:
                    logger.info(f"[AUTO LLM TRIGGER] Found project_id: {project_id}")
                    
                    # Import the multi-video manager to check project status
                    from app.services.multi_video_manager import get_multi_video_manager
                    
                    # Check if all analysis jobs are completed
                    multi_video_manager = await get_multi_video_manager()
                    project = await multi_video_manager.get_project(UUID(project_id))
                    
                    if project:
                        # Count completed analysis jobs
                        completed_analysis_jobs = 0
                        total_analysis_jobs = len(project.analysis_jobs)
                        
                        for analysis_job_id in project.analysis_jobs:
                            analysis_job = self._load_job_from_redis(analysis_job_id)
                            if analysis_job and analysis_job.status == ProcessingStatus.COMPLETED:
                                completed_analysis_jobs += 1
                        
                        logger.info(f"[AUTO LLM TRIGGER] Analysis jobs: {completed_analysis_jobs}/{total_analysis_jobs} completed")
                        
                        # Only trigger LLM recommendation if all analysis jobs are completed
                        if completed_analysis_jobs == total_analysis_jobs:
                            logger.info(f"[AUTO LLM TRIGGER] All analysis jobs completed, triggering LLM recommendation...")
                            
                            # Import the routes module to call the LLM recommendation
                            from app.api.routes import _trigger_llm_recommendation_after_cross_analysis
                            
                            # Trigger LLM recommendation asynchronously
                            import asyncio
                            asyncio.create_task(_trigger_llm_recommendation_after_cross_analysis(project_id))
                            
                            logger.info(f"[AUTO LLM TRIGGER] LLM recommendation triggered for project {project_id}")
                        else:
                            logger.info(f"[AUTO LLM TRIGGER] Waiting for all analysis jobs to complete before triggering LLM recommendation")
                    else:
                        logger.warning(f"[AUTO LLM TRIGGER] Project {project_id} not found")
                else:
                    logger.warning(f"[AUTO LLM TRIGGER] No project_id found in cross-analysis job metadata")
                    
            except Exception as e:
                logger.error(f"[AUTO LLM TRIGGER] Failed to check analysis jobs or trigger LLM recommendation: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"[EDIT JOB ERROR] Editing failed for job {job_id}: {e}")
            # Try to update job status
            try:
                job = self._load_job_from_redis(job_id)
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    self._save_job_to_redis(job)
            except Exception as inner:
                logger.error(f"[EDIT JOB ERROR] Could not update job status for {job_id}: {inner}")
            return False

    async def _process_cross_analysis_job(self, job_id: UUID) -> bool:
        """Process cross-video analysis job."""
        logger.info(f"[CROSS ANALYSIS JOB START] _process_cross_analysis_job called with job_id={job_id}")
        
        try:
            # Load job from Redis
            job = self._load_job_from_redis(ensure_uuid(job_id))
            if not job:
                logger.error(f"[CROSS ANALYSIS JOB ERROR] Job {job_id} not found in Redis")
                return False
            
            logger.info(f"[CROSS ANALYSIS JOB] Processing cross-analysis for job {job_id}")
            
            # Update job status
            job.status = ProcessingStatus.EDITING
            job.progress = 10
            self._save_job_to_redis(job)
            
            # Get project information
            project_id = job.metadata.get("project_id")
            if not project_id:
                logger.error(f"[CROSS ANALYSIS JOB ERROR] No project_id found in job metadata")
                return False
            
            # Load project
            from app.services.multi_video_manager import MultiVideoProjectManager
            from uuid import UUID
            project_uuid = UUID(project_id)
            multi_video_manager = MultiVideoProjectManager()
            project = await multi_video_manager.get_project(project_uuid)
            
            if not project:
                logger.error(f"[CROSS ANALYSIS JOB ERROR] Project {project_id} not found")
                return False
            
            # Update job progress
            job.progress = 50
            self._save_job_to_redis(job)
            
            # Perform cross-video analysis
            # This is a placeholder - implement actual cross-video analysis logic
            logger.info(f"[CROSS ANALYSIS JOB] Cross-video analysis completed for project {project_id}")
            
            # Update job status
            job.status = ProcessingStatus.COMPLETED
            job.progress = 100
            self._save_job_to_redis(job)
            
            # Update project status
            await multi_video_manager.get_project_status(project_uuid)
            
            return True
            
        except Exception as e:
            logger.error(f"[CROSS ANALYSIS JOB ERROR] Cross-video analysis failed for job {job_id}: {e}")
            # Try to update job status
            try:
                job = self._load_job_from_redis(job_id)
                if job:
                    job.status = ProcessingStatus.FAILED
                    job.error_message = str(e)
                    self._save_job_to_redis(job)
            except Exception as inner:
                logger.error(f"[CROSS ANALYSIS JOB ERROR] Could not update job status for {job_id}: {inner}")
            return False


# Global job queue instance
_job_queue: Optional[JobQueue] = None


async def initialize_redis_connection(redis_url: str) -> JobQueue:
    """
    Initialize the global job queue.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        JobQueue: Initialized job queue
    """
    global _job_queue
    
    settings = get_settings()
    
    _job_queue = JobQueue(
        redis_url=redis_url,
        queue_name=settings.queue_name
    )
    
    # Test connection
    if not await _job_queue.health_check():
        raise ConnectionError("Failed to connect to Redis")
    
    logger.info("Job queue initialized successfully")
    return _job_queue


async def get_job_queue() -> JobQueue:
    """
    Get the global job queue instance.
    
    Returns:
        JobQueue: Job queue instance
        
    Raises:
        RuntimeError: If job queue is not initialized
    """
    if _job_queue is None:
        raise RuntimeError("Job queue not initialized. Call initialize_redis_connection first.")
    
    return _job_queue 

def ensure_uuid(val):
    if isinstance(val, UUID):
        return val
    return UUID(str(val))


# Standalone function for RQ job processing

def process_analysis_job_standalone(job_id_str: str):
    """Standalone function for processing analysis jobs (for RQ worker)."""
    import asyncio
    import logging
    from uuid import UUID
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    job_id = UUID(job_id_str)
    logger.info(f"[ANALYSIS] Starting analysis job {job_id}")
    
    try:
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Import and initialize job queue
        from app.job_queue.worker import initialize_redis_connection
        from app.config.settings import settings
        
        async def run_analysis():
            # Initialize the job queue
            await initialize_redis_connection(settings.redis_url)
            from app.job_queue.worker import get_job_queue
            job_queue = await get_job_queue()
            return await job_queue._process_analysis_job(job_id)
        
        # Process the analysis job
        result = loop.run_until_complete(run_analysis())
        
        logger.info(f"[ANALYSIS] Analysis completed for job {job_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[ANALYSIS] Exception in analysis for job {job_id}: {e}")
        return False
    finally:
        loop.close()


def process_editing_job_standalone(job_id_str: str):
    """Standalone function for processing editing jobs (for RQ worker)."""
    import asyncio
    import logging
    from uuid import UUID
    import signal
    import threading
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    job_id = UUID(job_id_str)
    logger.info(f"[EDITING] Starting editing job {job_id}")
    
    # Global flag for graceful shutdown
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"[EDITING] Received signal {signum}, initiating graceful shutdown")
        shutdown_event.set()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create event loop with proper cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set up loop exception handler
        def loop_exception_handler(loop, context):
            logger.error(f"[EDITING] Event loop exception: {context}")
        
        loop.set_exception_handler(loop_exception_handler)
        
        # Import and initialize job queue
        from app.job_queue.worker import initialize_redis_connection
        from app.config.settings import settings
        
        async def run_editing():
            # Check for shutdown signal
            if shutdown_event.is_set():
                logger.info(f"[EDITING] Shutdown requested, aborting job {job_id}")
                return False
                
            # Initialize the job queue
            await initialize_redis_connection(settings.redis_url)
            from app.job_queue.worker import get_job_queue
            job_queue = await get_job_queue()
            
            # Process the editing job with timeout and shutdown monitoring
            try:
                # Create a task that can be cancelled
                editing_task = asyncio.create_task(job_queue._process_editing_job(job_id))
                
                # Monitor for shutdown while processing
                while not editing_task.done():
                    if shutdown_event.is_set():
                        logger.info(f"[EDITING] Shutdown requested, cancelling job {job_id}")
                        editing_task.cancel()
                        break
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
                if editing_task.cancelled():
                    return False
                
                return await editing_task
                
            except asyncio.CancelledError:
                logger.info(f"[EDITING] Job {job_id} was cancelled")
                return False
        
        # Process the editing job with proper timeout handling
        try:
            # Set a reasonable timeout for the entire operation
            result = loop.run_until_complete(asyncio.wait_for(run_editing(), timeout=3600))  # 1 hour timeout
            logger.info(f"[EDITING] Editing completed for job {job_id}: {result}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[EDITING] Job {job_id} timed out after 1 hour")
            # Clean up stuck FFmpeg processes before returning
            _kill_stuck_ffmpeg_processes()
            return False
        
    except Exception as e:
        logger.error(f"[EDITING] Exception in editing for job {job_id}: {e}")
        import traceback
        logger.error(f"[EDITING] Traceback: {traceback.format_exc()}")
        # Clean up stuck FFmpeg processes before returning
        _kill_stuck_ffmpeg_processes()
        return False
    finally:
        # Ensure proper cleanup
        try:
            # Cancel any pending tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending_tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            if pending_tasks:
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            
            # Close the loop
            loop.close()
        except Exception as cleanup_error:
            logger.error(f"[EDITING] Error during cleanup: {cleanup_error}")


def process_cross_analysis_job_standalone(job_id_str: str):
    """Process cross-video analysis job."""
    import asyncio
    import logging
    from uuid import UUID
    import signal
    import threading
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    job_id = UUID(job_id_str)
    logger.info(f"[CROSS ANALYSIS] Starting cross-video analysis for job {job_id}")
    
    # Global flag for graceful shutdown
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"[CROSS ANALYSIS] Received signal {signum}, initiating graceful shutdown")
        shutdown_event.set()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create event loop with proper cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set up loop exception handler
        def loop_exception_handler(loop, context):
            logger.error(f"[CROSS ANALYSIS] Event loop exception: {context}")
        
        loop.set_exception_handler(loop_exception_handler)
        
        # Import job queue
        from app.job_queue.worker import get_job_queue
        
        async def run_cross_analysis():
            # Check for shutdown signal
            if shutdown_event.is_set():
                logger.info(f"[CROSS ANALYSIS] Shutdown requested, aborting job {job_id}")
                return False
                
            job_queue = await get_job_queue()
            
            # Process the cross-analysis job with timeout and shutdown monitoring
            try:
                # Create a task that can be cancelled
                analysis_task = asyncio.create_task(job_queue._process_cross_analysis_job(job_id))
                
                # Monitor for shutdown while processing
                while not analysis_task.done():
                    if shutdown_event.is_set():
                        logger.info(f"[CROSS ANALYSIS] Shutdown requested, cancelling job {job_id}")
                        analysis_task.cancel()
                        break
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
                if analysis_task.cancelled():
                    return False
                
                return await analysis_task
                
            except asyncio.CancelledError:
                logger.info(f"[CROSS ANALYSIS] Job {job_id} was cancelled")
                return False
        
        # Process the cross-analysis job with proper timeout handling
        try:
            # Set a reasonable timeout for the entire operation
            result = asyncio.wait_for(run_cross_analysis(), timeout=1800)  # 30 minute timeout
            logger.info(f"[CROSS ANALYSIS] Cross-video analysis completed for job {job_id}: {result}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[CROSS ANALYSIS] Job {job_id} timed out after 30 minutes")
            return False
        
    except Exception as e:
        logger.error(f"[CROSS ANALYSIS] Exception in cross-video analysis for job {job_id}: {e}")
        import traceback
        logger.error(f"[CROSS ANALYSIS] Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Ensure proper cleanup
        try:
            # Cancel any pending tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending_tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            if pending_tasks:
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            
            # Close the loop
            loop.close()
        except Exception as cleanup_error:
            logger.error(f"[CROSS ANALYSIS] Error during cleanup: {cleanup_error}")

def process_editing_job_subprocess(job_id_str: str):
    """
    Subprocess-based function for processing editing jobs.
    This function runs the editing in a subprocess to isolate segmentation faults.
    
    Args:
        job_id_str: Job ID as string (RQ can't pickle UUIDs directly)
    """
    import subprocess
    import tempfile
    import os
    import json
    from uuid import UUID
    
    job_id = UUID(job_id_str)
    logger.info(f"[SUBPROCESS EDITING] Starting editing job {job_id} in subprocess")
    
    try:
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_out:
            output_json_path = tmp_out.name
        
        # Run editing in subprocess with enhanced shader integration
        script_path = os.path.abspath('run_editing_subprocess_dynamic.py')
        logger.info(f"[SUBPROCESS EDITING] Running enhanced subprocess with shader integration: {sys.executable} {script_path} {job_id} {output_json_path}")
        subprocess_result = subprocess.run([
            sys.executable, script_path, str(job_id), output_json_path
        ], capture_output=True, text=True, timeout=1800, cwd=os.path.dirname(script_path))  # 30 minute timeout
        
        if subprocess_result.returncode != 0:
            logger.error(f"Editing subprocess failed with return code {subprocess_result.returncode}")
            logger.error(f"Subprocess stdout: {subprocess_result.stdout}")
            logger.error(f"Subprocess stderr: {subprocess_result.stderr}")
            raise RuntimeError(f"Editing subprocess failed: {subprocess_result.stderr}")
        
        # Read results from JSON file
        if not os.path.exists(output_json_path):
            raise RuntimeError("Editing output file not found")
        
        with open(output_json_path, 'r') as f:
            result_data = json.load(f)
        
        # Clean up temporary file
        os.remove(output_json_path)
        
        # Check for errors in the result
        if not result_data.get("success", False):
            error_msg = result_data.get("error", "Unknown error")
            raise RuntimeError(f"Editing failed: {error_msg}")
        
        # Log progress
        for log_entry in result_data.get("progress_log", []):
            logger.info(f"[SUBPROCESS EDITING] {log_entry}")
        
        logger.info(f"[SUBPROCESS EDITING] Completed editing job {job_id}")
        
    except subprocess.TimeoutExpired:
        logger.error(f"[SUBPROCESS EDITING] Editing subprocess timed out")
        # Try to update job status
        try:
            job_queue = get_job_queue()
            job = job_queue._load_job_from_redis(job_id)
            if job:
                job.status = ProcessingStatus.FAILED
                job.error_message = "Editing subprocess timed out"
                job_queue._save_job_to_redis(job)
        except Exception as inner:
            logger.error(f"[SUBPROCESS EDITING] Could not update job status for {job_id}: {inner}")
    except Exception as e:
        logger.error(f"[SUBPROCESS EDITING] Failed to process editing job {job_id}: {e}")
        # Try to update job status
        try:
            job_queue = get_job_queue()
            job = job_queue._load_job_from_redis(job_id)
            if job:
                job.status = ProcessingStatus.FAILED
                job.error_message = str(e)
                job_queue._save_job_to_redis(job)
        except Exception as inner:
            logger.error(f"[SUBPROCESS EDITING] Could not update job status for {job_id}: {inner}") 

def safe_set_job_metadata(job: ProcessingJob, field: str, value: Any) -> None:
    """
    Safely set job metadata field with type validation.
    
    Args:
        job: The processing job
        field: Field name to set
        value: Value to set
    """
    try:
        if hasattr(job, field):
            setattr(job, field, value)
        else:
            # Store in metadata if field doesn't exist
            if not hasattr(job, 'metadata'):
                job.metadata = {}
            job.metadata[field] = value
            logger.info(f"[METADATA] Stored {field} in job metadata")
    except Exception as e:
        logger.error(f"[METADATA] Failed to set {field}: {e}")
        # Fallback to metadata storage
        if not hasattr(job, 'metadata'):
            job.metadata = {}
        job.metadata[field] = value

def _validate_job_schema(job: ProcessingJob) -> bool:
    """
    Validate job schema before saving to Redis.
    
    Args:
        job: The processing job to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Validate the job schema
        job_dict = job.dict()
        # Re-create the job to trigger validation
        ProcessingJob(**job_dict)
        return True
    except ValidationError as e:
        logger.error(f"[VALIDATION] Job schema validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"[VALIDATION] Unexpected validation error: {e}")
        return False

async def close_redis_connection():
    """Close Redis connection on shutdown."""
    global _job_queue
    if _job_queue:
        try:
            _job_queue.redis_conn.close()
            _job_queue = None
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

# Export standalone functions for RQ
def start_worker():
    """Start the RQ worker for processing background jobs."""
    import asyncio
    from rq import Worker, Queue
    from redis import Redis
    
    logger.info("👷 Starting RQ Worker...")
    
    try:
        # Get settings
        from app.config.settings import settings
        
        # Connect to Redis (simple connection, no complex initialization)
        redis_conn = Redis.from_url(settings.redis_url)
        redis_conn.ping()
        logger.info(f"✅ Redis connection established to {settings.redis_url}")
        
        # Initialize global job queue for standalone functions
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from app.job_queue.worker import initialize_redis_connection
            global _job_queue
            _job_queue = loop.run_until_complete(initialize_redis_connection(settings.redis_url))
            logger.info("✅ Global job queue initialized for standalone functions")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize global job queue: {e}")
        finally:
            loop.close()
        
        # Create queue
        queue = Queue(settings.queue_name, connection=redis_conn)
        logger.info(f"📦 Queue '{settings.queue_name}' ready")
        
        # Create and start worker
        worker = Worker([queue], connection=redis_conn)
        logger.info(f"👷 Worker '{worker.name}' created for queue: {settings.queue_name}")
        logger.info("🏃 Worker running - waiting for jobs...")
        
        # Start working
        worker.work()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Worker stopped by user")
    except Exception as e:
        logger.error(f"💥 Worker error: {e}")
        raise

__all__ = [
    'process_analysis_job_standalone',
    'process_editing_job_standalone',
    'process_cross_analysis_job_standalone',
    'process_editing_job_subprocess',
    'get_job_queue',
    'JobQueue',
    'start_worker'
] 