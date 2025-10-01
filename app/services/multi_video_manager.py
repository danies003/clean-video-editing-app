"""
Multi-Video Project Manager
Handles multi-video projects, cross-video analysis, and combined editing.
"""

import logging
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass

from app.models.schemas import (
    MultiVideoProject, 
    CrossVideoAnalysisResult,
    ProcessingStatus
)
from app.services.manager import get_service_manager
# Import job_queue only when needed to avoid circular import

logger = logging.getLogger(__name__)

@dataclass
class MultiVideoProjectManager:
    """Manages multi-video projects and their lifecycle."""
    
    def __post_init__(self):
        self.redis_client = None
        self.job_queue = None
    
    async def initialize(self):
        """Initialize the manager with Redis and job queue."""
        service_manager = get_service_manager()
        job_queue = await service_manager.get_redis()
        self.redis_client = job_queue.redis_conn
        logger.info(f"Initialized multi-video manager with Redis client: {self.redis_client is not None}")
        
        # Test Redis connection
        try:
            test_result = self.redis_client.set("test_init", "test_value")
            logger.info(f"Redis connection test result: {test_result}")
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            raise
        
        # Import job_queue locally to avoid circular import
        from app.job_queue.worker import get_job_queue
        self.job_queue = await get_job_queue()
    
    async def _ensure_initialized(self):
        if self.redis_client is None:
            logger.warning("[MGR INIT] Redis client was None, initializing now...")
            await self.initialize()

    async def create_project(self, name: str, video_ids: List[UUID]) -> MultiVideoProject:
        await self._ensure_initialized()
        project_id = uuid4()
        project = MultiVideoProject(
            project_id=project_id,
            name=name,
            video_ids=video_ids,
            status=ProcessingStatus.PENDING,
            analysis_jobs=[],
            cross_analysis_job=None,
            editing_job=None,
            output_video_id=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )
        await self._save_project(project)
        logger.info(f"Created multi-video project {project_id} with {len(video_ids)} videos")
        return project
    
    async def get_project(self, project_id: UUID) -> Optional[MultiVideoProject]:
        await self._ensure_initialized()
        project_key = f"multi_video_project:{project_id}"
        try:
            project_data = self.redis_client.get(project_key)
            logger.info(f"Retrieved project data for {project_id}: {project_data is not None}")
            
            if project_data:
                import json
                data = json.loads(project_data.decode('utf-8'))
                logger.info(f"Loaded project data: {data}")
                logger.info(f"Raw editing_job from data: {data.get('editing_job')}")
                logger.info(f"Type of editing_job: {type(data.get('editing_job'))}")
                project = MultiVideoProject(**data)
                logger.info(f"Created project object with editing_job: {project.editing_job}")
                logger.info(f"Type of project.editing_job: {type(project.editing_job)}")
                return project
            else:
                logger.warning(f"Project {project_id} not found in Redis")
                return None
        except Exception as e:
            logger.error(f"Failed to get project {project_id} from Redis: {e}")
            raise
    
    async def update_project(self, project: MultiVideoProject):
        await self._ensure_initialized()
        project.updated_at = datetime.utcnow()
        await self._save_project(project)
    
    async def _save_project(self, project: MultiVideoProject):
        await self._ensure_initialized()
        project_key = f"multi_video_project:{project.project_id}"
        project_data = project.model_dump_json()
        try:
            # Test Redis connection first
            test_result = self.redis_client.set("test_key", "test_value")
            logger.info(f"Redis test result: {test_result}")
            
            # Save the project
            result = self.redis_client.set(project_key, project_data)
            logger.info(f"Saved project {project.project_id} to Redis: {result}")
        except Exception as e:
            logger.error(f"Failed to save project {project.project_id} to Redis: {e}")
            raise
    
    async def add_analysis_job(self, project_id: UUID, analysis_job_id: UUID):
        await self._ensure_initialized()
        logger.info(f"[ADD ANALYSIS JOB] Attempting to add analysis job {analysis_job_id} to project {project_id}")
        
        project = await self.get_project(project_id)
        if project:
            logger.info(f"[ADD ANALYSIS JOB] Found project {project_id}, current analysis_jobs: {project.analysis_jobs}")
            
            if analysis_job_id not in project.analysis_jobs:
                project.analysis_jobs.append(analysis_job_id)
                await self.update_project(project)
                logger.info(f"[ADD ANALYSIS JOB] Successfully added analysis job {analysis_job_id} to project {project_id}")
                logger.info(f"[ADD ANALYSIS JOB] Updated analysis_jobs: {project.analysis_jobs}")
            else:
                logger.info(f"[ADD ANALYSIS JOB] Analysis job {analysis_job_id} already exists in project {project_id}")
        else:
            logger.error(f"[ADD ANALYSIS JOB] Project {project_id} not found")
    
    async def set_cross_analysis_job(self, project_id: UUID, cross_analysis_job_id: UUID):
        await self._ensure_initialized()
        logger.info(f"Setting cross-analysis job {cross_analysis_job_id} for project {project_id}")
        project = await self.get_project(project_id)
        if project:
            logger.info(f"Found project {project_id}, updating cross-analysis job")
            project.cross_analysis_job = cross_analysis_job_id
            await self.update_project(project)
            logger.info(f"Set cross-analysis job {cross_analysis_job_id} for project {project_id}")
        else:
            logger.error(f"Project {project_id} not found when trying to set cross-analysis job")
            raise Exception(f"Project {project_id} not found")
    
    async def set_editing_job(self, project_id: UUID, editing_job_id: UUID):
        await self._ensure_initialized()
        project = await self.get_project(project_id)
        if project:
            project.editing_job = editing_job_id
            await self.update_project(project)
            logger.info(f"Set editing job {editing_job_id} for project {project_id}")
    
    async def get_project_status(self, project_id: UUID) -> Optional[MultiVideoProject]:
        await self._ensure_initialized()
        project = await self.get_project(project_id)
        if not project:
            return None
        
        # Get the job queue instance
        job_queue = self.job_queue
        
        # Skip analysis and cross-analysis phases - new workflow uses create_robust_25_second_video.py directly
        analysis_completed = len(project.video_ids)  # Assume all analysis is completed
        cross_analysis_completed = True  # Skip cross-analysis phase
        
        # Check editing status and progress
        editing_completed = False
        editing_progress = 0
        if project.editing_job:
            edit_job = job_queue._load_job_from_redis(project.editing_job)
            if edit_job:
                editing_completed = edit_job.status == ProcessingStatus.COMPLETED
                editing_progress = edit_job.progress if hasattr(edit_job, 'progress') else 0
        
        # Calculate overall progress - skip analysis and cross-analysis phases
        # Use the actual job progress instead of just completed/not completed
        progress = editing_progress
        
        # Update project status and progress - skip analysis and cross-analysis phases
        if editing_completed:
            project.status = ProcessingStatus.COMPLETED
            project.progress = 100.0
        else:
            # Move directly to editing phase since analysis and cross-analysis are skipped
            project.status = ProcessingStatus.EDITING
            project.progress = progress
        
        await self.update_project(project)
        return project
    
    async def list_projects(self) -> List[MultiVideoProject]:
        """List all multi-video projects."""
        await self._ensure_initialized()
        projects = []
        
        try:
            # Get all keys that match the multi-video project pattern
            pattern = "multi_video_project:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    project_data = self.redis_client.get(key)
                    if project_data:
                        import json
                        data = json.loads(project_data.decode('utf-8'))
                        project = MultiVideoProject(**data)
                        projects.append(project)
                except Exception as e:
                    logger.error(f"Failed to load project from key {key}: {e}")
                    continue
            
            logger.info(f"Found {len(projects)} multi-video projects")
            return projects
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    async def update_project_status(self, project_id: UUID, status: str, output_video_id: Optional[UUID] = None):
        """Update project status and optionally set output video ID."""
        await self._ensure_initialized()
        project = await self.get_project(project_id)
        if project:
            project.status = ProcessingStatus(status)
            if output_video_id:
                project.output_video_id = output_video_id
            project.updated_at = datetime.utcnow()
            await self.update_project(project)
            logger.info(f"Updated project {project_id} status to {status}")
        else:
            logger.error(f"Project {project_id} not found when trying to update status")
            raise Exception(f"Project {project_id} not found")

# Global instance
_multi_video_manager = None

async def get_multi_video_manager() -> MultiVideoProjectManager:
    """Get the global multi-video project manager instance."""
    global _multi_video_manager
    if _multi_video_manager is None:
        _multi_video_manager = MultiVideoProjectManager()
        await _multi_video_manager.initialize()
    return _multi_video_manager 