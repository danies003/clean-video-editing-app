"""
FastAPI route definitions for the video editing automation engine.

This module contains all API endpoints organized by functionality,
including video upload, analysis, editing, template management, and health checks.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query, Body, Form
from fastapi.responses import JSONResponse, StreamingResponse

from app.models.schemas import (
    VideoUploadRequest, VideoUploadResponse, VideoAnalysisResult,
    AnalyzeVideoRequest, EditVideoRequest, ProcessingJob, JobStatusResponse,
    JobListResponse, EditingTemplate, TemplateListResponse, HealthCheckResponse,
    BaseResponse, ErrorResponse, ProcessingStatus, TemplateType, AdvancedEditRequest, EditDecisionMap,
    MultiVideoUploadRequest, MultiVideoUploadResponse,
    CrossVideoAnalysisRequest, CrossVideoAnalysisResult,
    MultiVideoEditRequest, MultiVideoEditResponse,
    MultiVideoProjectStatus, QualityPreset, EditStyle,
    SegmentRecommendations, MoviePyRenderingPlan, MultiVideoProject,
    MultiVideoAnalysisRequest, BeatDetectionResult, MotionAnalysisResult, AudioAnalysisResult
)
from app.ingestion.storage import get_storage_client
# Remove top-level import of heavy dependencies - import them only when needed
# from app.analyzer.engine import get_analysis_engine
from app.templates.manager import get_template_manager
from app.timeline.builder import get_timeline_builder
from app.editor.renderer import get_renderer
from app.job_queue.worker import get_job_queue
from app.editor.advanced_edit import generate_advanced_edit_plan, edit_decision_map_to_template
import traceback
from app.services.multi_video_manager import get_multi_video_manager
from app.editor.enhanced_llm_editor import get_enhanced_llm_editor
from app.api.auth import router as auth_router
from app.api.music_routes import router as music_router
from app.services.redis_validator import RedisValidator
import redis

logger = logging.getLogger(__name__)

# Create routers
health_router = APIRouter()
video_router = APIRouter()
template_router = APIRouter()

# Multi-Video Project Router
multi_video_router = APIRouter(prefix="/multi-video", tags=["Multi-Video Projects"])


# Health Check Routes
@health_router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        HealthCheckResponse: Service health status and metadata
    """
    import time
    start_time = getattr(health_check, '_start_time', time.time())
    
    # Simple health check - just return healthy
    return HealthCheckResponse(
        status="healthy",
        services={"api": "healthy"},
        uptime=time.time() - start_time
    )

@health_router.get("", response_model=HealthCheckResponse)
async def health_check_no_slash():
    """
    Health check endpoint without trailing slash to prevent redirects.
    """
    import time
    start_time = getattr(health_check, '_start_time', time.time())
    
    return HealthCheckResponse(
        status="healthy",
        services={"api": "healthy"},
        uptime=time.time() - start_time
    )

@health_router.get("/simple")
async def simple_health_check():
    """
    Simple health check that doesn't depend on any services.
    """
    return {"status": "healthy", "message": "Server is running"}

@health_router.get("/services")
async def services_health_check():
    """
    Detailed health check for all services with graceful degradation.
    """
    from app.services import get_service_manager
    
    try:
        service_manager = get_service_manager()
        health_status = await service_manager.health_check()
        return {
            "status": "healthy",
            "services": health_status,
            "message": "Service health check completed"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "services": {"core": "healthy", "error": str(e)},
            "message": "Health check failed but core is running"
        }

@health_router.get("/diagnostic")
async def environment_diagnostic():
    """
    Comprehensive environment diagnostic endpoint.
    Returns detailed information about the environment for debugging.
    """
    import sys
    import os
    import platform
    import subprocess
    from pathlib import Path
    
    diagnostic_info = {
        "environment": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "environment_variables": dict(os.environ),
        },
        "dependencies": {},
        "imports": {},
        "file_system": {},
        "process": {
            "pid": os.getpid(),
            "user": os.getenv("USER", "unknown"),
            "home": os.getenv("HOME", "unknown"),
        }
    }
    
    # Test critical imports
    critical_imports = [
        "fastapi", "uvicorn", "redis", "boto3", 
        "cv2", "librosa", "numpy", "moviepy", "opencv-python"
    ]
    
    for module in critical_imports:
        try:
            if module == "cv2":
                import cv2
                diagnostic_info["imports"][module] = {
                    "status": "success",
                    "version": getattr(cv2, "__version__", "unknown")
                }
            elif module == "librosa":
                import librosa
                diagnostic_info["imports"][module] = {
                    "status": "success",
                    "version": getattr(librosa, "__version__", "unknown")
                }
            elif module == "numpy":
                import numpy
                diagnostic_info["imports"][module] = {
                    "status": "success",
                    "version": numpy.__version__
                }
            elif module == "moviepy":
                import moviepy
                diagnostic_info["imports"][module] = {
                    "status": "success",
                    "version": moviepy.__version__
                }
            else:
                imported_module = __import__(module)
                diagnostic_info["imports"][module] = {
                    "status": "success",
                    "version": getattr(imported_module, "__version__", "unknown")
                }
        except ImportError as e:
            diagnostic_info["imports"][module] = {
                "status": "failed",
                "error": str(e)
            }
        except Exception as e:
            diagnostic_info["imports"][module] = {
                "status": "error",
                "error": str(e)
            }
    
    # Check file system
    try:
        diagnostic_info["file_system"]["requirements_exists"] = Path("requirements.txt").exists()
        diagnostic_info["file_system"]["main_py_exists"] = Path("main.py").exists()
        diagnostic_info["file_system"]["start_py_exists"] = Path("start.py").exists()
        diagnostic_info["file_system"]["app_dir_exists"] = Path("app").exists()
    except Exception as e:
        diagnostic_info["file_system"]["error"] = str(e)
    
    # Check pip list
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True, timeout=10)
        diagnostic_info["dependencies"]["pip_list"] = result.stdout
    except Exception as e:
        diagnostic_info["dependencies"]["pip_list_error"] = str(e)
    
    return diagnostic_info

@health_router.get("/health/redis")
async def redis_health_check():
    """Check Redis health and detect corruption."""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        validator = RedisValidator(redis_client)
        
        health_status = validator.validate_redis_health()
        
        return {
            "status": "success",
            "redis_health": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Video Processing Routes
@video_router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    request: VideoUploadRequest,
    storage_client=Depends(get_storage_client)
):
    """
    Initiate video upload process.
    
    Creates a pre-signed URL for direct upload to cloud storage.
    
    Args:
        request: Video upload request with metadata
        storage_client: Storage client dependency
        
    Returns:
        VideoUploadResponse: Upload URL and expiration details
    """
    try:
        # If a video_url is provided, treat it as the upload location
        if request.video_url:
            from uuid import uuid4
            from datetime import datetime, timedelta
            # Use the provided video_url as the upload_url, set expires_at far in the future
            return VideoUploadResponse(
                video_id=str(uuid4()),
                upload_url=request.video_url,
                expires_at=(datetime.utcnow() + timedelta(days=3650)).isoformat()  # 10 years in the future
            )
        # Validate file size
        max_size = 500 * 1024 * 1024  # 500MB
        if request.file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size {request.file_size} exceeds maximum allowed size of {max_size}"
            )
        # Validate file format
        file_extension = request.filename.split('.')[-1].lower()
        supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'wmv']
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. Supported formats: {supported_formats}"
            )
        # Generate upload URL
        upload_url, expires_at = await storage_client.create_upload_url(
            filename=request.filename,
            content_type=request.content_type,
            metadata={
                "template_type": request.template_type.value if request.template_type else "beat_match",
                "quality_preset": request.quality_preset.value,
                "custom_settings": request.custom_settings
            }
        )
        from uuid import uuid4
        return VideoUploadResponse(
            video_id=str(uuid4()),
            upload_url=upload_url,
            expires_at=expires_at.isoformat()
        )
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ [API] Video upload initiation failed")
        logger.error(f"❌ [API] Exception type: {type(e).__name__}")
        logger.error(f"❌ [API] Exception message: {str(e)}")
        logger.error(f"❌ [API] Full traceback:")
        logger.error(tb)
        logger.error(f"❌ [API] Request context:")
        logger.error(f"   - Filename: {request.filename}")
        logger.error(f"   - File size: {request.file_size}")
        logger.error(f"   - Content type: {request.content_type}")
        logger.error(f"   - Template type: {request.template_type}")
        logger.error(f"   - Quality preset: {request.quality_preset}")
        raise HTTPException(status_code=500, detail="Failed to initiate video upload")


@video_router.post("/upload-direct", response_model=VideoUploadResponse)
async def upload_video_direct(
    file: UploadFile = File(...)
):
    from app.services import get_service_manager
    storage_client = await get_service_manager().get_storage()
    """
    Direct video file upload endpoint with automatic format conversion.
    
    Accepts any video file format and automatically converts it to browser-compatible MP4.
    
    Args:
        file: The video file to upload
        storage_client: Storage client dependency
        
    Returns:
        VideoUploadResponse: Video ID and storage details
    """
    import tempfile
    import os
    try:
        from uuid import uuid4
        from datetime import datetime, timedelta
        
        # Accept ANY video file format - we'll convert it automatically
        valid_video_extensions = {
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.3gp', 
            '.ogv', '.ts', '.mts', '.m2ts', '.vob', '.asf', '.rm', '.rmvb', '.f4v'
        }
        is_video_content_type = file.content_type and file.content_type.startswith('video/')
        is_video_extension = file.filename and any(file.filename.lower().endswith(ext) for ext in valid_video_extensions)
        
        if not (is_video_content_type or is_video_extension):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be a video. Got: {file.content_type} for {file.filename}. Supported formats: {', '.join(valid_video_extensions)}"
            )
        
        # Generate unique video ID
        video_id = uuid4()
        
        # Create temporary file for original upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            # Read and write file content to temporary file
            file_content = await file.read()
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        converted_file_path = None
        try:
            # Import video converter
            from app.ingestion.video_converter import video_converter
            
            # Convert video to browser-compatible format
            logger.info(f"🎬 Converting video to browser-compatible format: {file.filename}")
            logger.info(f"📊 Original file size: {len(file_content)} bytes")
            logger.info(f"📊 Original content type: {file.content_type}")
            
            converted_file_path, video_info = await video_converter.convert_video(
                input_path=temp_file_path,
                quality_preset="medium"
            )
            
            logger.info(f"✅ Video conversion completed successfully")
            logger.info(f"📁 Converted file path: {converted_file_path}")
            logger.info(f"📊 Converted file size: {os.path.getsize(converted_file_path) if converted_file_path else 'unknown'} bytes")
            
            # Generate storage key for converted video (always MP4)
            storage_key = f"uploads/{video_id}.mp4"
            
            # Upload converted file using storage client
            # Reduce metadata size to avoid S3 limits
            simplified_video_info = {
                'format': video_info.get('format', {}).get('format_name', 'unknown'),
                'duration': video_info.get('format', {}).get('duration', '0'),
                'size': video_info.get('format', {}).get('size', '0'),
                'streams_count': len(video_info.get('streams', [])),
                'has_video': any(s.get('codec_type') == 'video' for s in video_info.get('streams', [])),
                'has_audio': any(s.get('codec_type') == 'audio' for s in video_info.get('streams', []))
            }
            
            logger.info(f"📤 Uploading converted video to S3: {storage_key}")
            logger.info(f"📊 Simplified video info: {simplified_video_info}")
            
            try:
                storage_url = await storage_client.upload_file(
                    file_path=converted_file_path,
                    file_key=storage_key,
                    content_type="video/mp4",  # Always MP4 after conversion
                    metadata={
                        'original_filename': file.filename or 'unknown',
                        'video_id': str(video_id),
                        'upload_timestamp': datetime.utcnow().isoformat(),
                        'converted_from': file.content_type,
                        'browser_compatible': 'true'
                    }
                )
                logger.info(f"✅ Video uploaded successfully to S3: {storage_url}")
            except Exception as upload_error:
                logger.error(f"❌ S3 upload failed: {upload_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to upload converted video: {str(upload_error)}"
                )
            
            # Get converted file size
            converted_file_size = os.path.getsize(converted_file_path) if converted_file_path else len(file_content)
            
            from pydantic import AnyUrl
            return VideoUploadResponse(
                video_id=str(video_id),
                upload_url=AnyUrl(str(storage_url)),
                expires_at=(datetime.utcnow() + timedelta(days=30)).isoformat(),
                metadata={
                    'filename': file.filename or 'unknown',
                    'original_size': len(file_content),
                    'converted_size': converted_file_size,
                    'original_content_type': file.content_type,
                    'converted_content_type': 'video/mp4',
                    'browser_compatible': True,
                    'video_info': simplified_video_info
                }
            )
            
        finally:
            # Clean up temporary files
            temp_files_to_cleanup = [temp_file_path]
            if converted_file_path and converted_file_path != temp_file_path:
                temp_files_to_cleanup.append(converted_file_path)
            
            await video_converter.cleanup_temp_files(temp_files_to_cleanup)
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ [API] Direct video upload failed")
        logger.error(f"❌ [API] Exception type: {type(e).__name__}")
        logger.error(f"❌ [API] Exception message: {str(e)}")
        logger.error(f"❌ [API] Full traceback:")
        logger.error(tb)
        logger.error(f"❌ [API] Upload context:")
        logger.error(f"   - Filename: {file.filename}")
        logger.error(f"   - Content type: {file.content_type}")
        logger.error(f"   - File size: {len(file_content) if 'file_content' in locals() else 'Unknown'}")
        logger.error(f"   - Temp file path: {temp_file_path if 'temp_file_path' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)} ({type(e).__name__})")


@video_router.post("/{video_id}/analyze", response_model=JobStatusResponse)
async def analyze_video(
    video_id: UUID,
    request: AnalyzeVideoRequest
):
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    """
    Trigger Gemini AI-powered video analysis for intelligent content understanding.
    
    Args:
        video_id: Unique identifier for the video
        request: Analysis configuration
        job_queue: Job queue dependency
        
    Returns:
        JobStatusResponse: Job status and metadata
    """
    try:
        # Import heavy dependencies only when needed
        from app.analyzer.engine import get_analysis_engine
        
        # Create Gemini analysis job
        job = await job_queue.create_analysis_job(
            video_id=video_id,
            template_type=request.template_type,
            analysis_options=request.analysis_options
        )
        # Store video S3 key in job metadata (not the download URL)
        # Extract video_id from the download URL to construct the S3 key
        video_id_from_url = str(video_id)
        job.metadata["video_url"] = f"uploads/{video_id_from_url}.mp4"
        job.metadata["analysis_type"] = "gemini"  # Mark as Gemini analysis
        # Queue the analysis job
        job_queue.enqueue_job(job)
        
        return JobStatusResponse(job=job)
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ [API] Gemini video analysis initiation failed for {video_id}")
        logger.error(f"❌ [API] Exception type: {type(e).__name__}")
        logger.error(f"❌ [API] Exception message: {str(e)}")
        logger.error(f"❌ [API] Full traceback:")
        logger.error(tb)
        logger.error(f"❌ [API] Analysis context:")
        logger.error(f"   - Video ID: {video_id}")
        logger.error(f"   - Template type: {request.template_type}")
        logger.error(f"   - Analysis options: {request.analysis_options}")
        logger.error(f"   - Video URL: {request.video_url}")
        raise HTTPException(status_code=500, detail="Failed to initiate Gemini video analysis")


@video_router.get("/{video_id}/status", response_model=JobStatusResponse)
async def get_video_status(
    video_id: UUID
):
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    """
    Get the current status of video processing.
    
    Args:
        video_id: Unique identifier for the video
        job_queue: Job queue dependency
        
    Returns:
        JobStatusResponse: Current job status and progress
    """
    try:
        job = await job_queue.get_job_by_video_id(video_id)
        if not job:
            raise HTTPException(status_code=404, detail="Video processing job not found")
        
        return JobStatusResponse(job=job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video status for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video status")


@video_router.post("/{video_id}/edit", response_model=JobStatusResponse)
async def edit_video(
    video_id: UUID,
    request: EditVideoRequest
):
    """
    Apply LLM-powered editing and start video rendering process.
    """
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    try:
        # Ensure analysis is completed before editing
        analysis_job = await job_queue.get_job_by_video_id(video_id)
        if not analysis_job or analysis_job.status != ProcessingStatus.COMPLETED or not analysis_job.analysis_result:
            raise HTTPException(status_code=400, detail="Analysis must be completed before editing.")
        
        # Create editing job with LLM metadata
        job = await job_queue.create_editing_job(
            video_id=video_id,
            template_id=request.template_id,
            template_type=request.template_type,
            custom_settings=request.custom_settings,
            quality_preset=request.quality_preset
        )
        
        # Add LLM editing metadata
        job.metadata["style"] = "tiktok"  # Default style
        job.metadata["edit_scale"] = 0.5  # Default edit scale
        
        # Queue the editing job
        job_queue.enqueue_job(job)
        return JobStatusResponse(job=job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video editing initiation failed for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate video editing")


@video_router.post("/{video_id}/preview")
async def preview_video(
    video_id: UUID,
    request: EditVideoRequest
):
    """
    Generate preview without full rendering - returns editing plan and preview data.
    This skips the heavy rendering process and provides immediate feedback.
    
    Args:
        video_id: Unique identifier for the video
        request: Edit configuration
        
    Returns:
        Dict containing preview data and editing plan
    """
    try:
        # Import heavy dependencies only when needed
        from app.analyzer.engine import get_analysis_engine
        from app.editor.llm_editor import LLMEditor, LLMProvider
        
        # Get analysis engine
        analysis_engine = get_analysis_engine()
        
        # Get video path from storage
        storage = get_storage_client()
        video_path = await storage.get_video_path(video_id)
        
        if not video_path:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Perform Gemini analysis
        gemini_analysis = await analysis_engine.analyze_video_with_gemini(video_path, video_id)
        
        # Create LLM editor with Gemini
        llm_editor = LLMEditor(provider=LLMProvider.GEMINI)
        
        # Convert Gemini analysis to VideoAnalysisResult format
        analysis_result = VideoAnalysisResult(
            video_id=video_id,
            duration=25.0,  # Default duration for preview
            fps=30.0,
            resolution=(1080, 1920),
            beat_detection=BeatDetectionResult(
                beats=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                bpm=120.0,
                confidence=0.8
            ),
            motion_analysis=MotionAnalysisResult(
                motion_spikes=[],
                scene_changes=[],
                confidence=0.8
            ),
            audio_analysis=AudioAnalysisResult(
                volume_levels=[],
                silence_periods=[],
                confidence=0.8
            )
        )
        
        # Generate editing plan using Gemini
        editing_plan = llm_editor.generate_editing_plan(
            analysis_result=analysis_result,
            style=request.style,
            target_duration=request.target_duration
        )
        
        # Generate MoviePy rendering plan
        rendering_plan = llm_editor.generate_moviepy_rendering_plan(
            analysis_result=analysis_result,
            style=request.style,
            target_duration=request.target_duration
        )
        
        # Return preview data
        return {
            "video_id": str(video_id),
            "gemini_analysis": gemini_analysis,
            "editing_plan": {
                "style": editing_plan.style,
                "target_duration": editing_plan.target_duration,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "effect": seg.effect,
                        "intensity": seg.intensity,
                        "reasoning": seg.reasoning
                    } for seg in editing_plan.segments
                ],
                "transitions": editing_plan.transitions,
                "effects": editing_plan.effects,
                "reasoning": editing_plan.reasoning,
                "confidence": editing_plan.confidence
            },
            "rendering_plan": {
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "effects": [{"name": eff.name, "params": eff.params} for eff in seg.effects],
                        "transitions": [{"name": trans.name, "params": trans.params} for trans in seg.transitions]
                    } for seg in rendering_plan.segments
                ]
            },
            "preview_ready": True,
            "message": "Preview generated successfully - ready for customization"
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ [API] Video preview generation failed for {video_id}")
        logger.error(f"❌ [API] Exception type: {type(e).__name__}")
        logger.error(f"❌ [API] Exception message: {str(e)}")
        logger.error(f"❌ [API] Full traceback:")
        logger.error(tb)
        raise HTTPException(status_code=500, detail="Failed to generate video preview")


@video_router.post("/{video_id}/advanced_edit")
async def advanced_edit(
    video_id: UUID,
    request: AdvancedEditRequest = Body(...)
):
    # Set the video_id from the URL path if not provided in body
    if request.video_id is None:
        request.video_id = video_id
    """
    Generate and render an advanced, stylized edit using existing analysis results.
    """
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    template_manager = await get_service_manager().get_template_manager()
    try:
        job = await job_queue.get_job_by_video_id(video_id)
        if not job or not job.analysis_result:
            raise HTTPException(status_code=404, detail="Analysis results not found for this video.")

        # Extract style from style_preferences or use default
        style = "tiktok"  # default
        if request.style_preferences:
            if "energy_level" in request.style_preferences:
                energy_level = request.style_preferences["energy_level"]
                if energy_level == "high":
                    style = "tiktok"
                elif energy_level == "medium":
                    style = "youtube"
                elif energy_level == "low":
                    style = "cinematic"
            
            # Override with pacing if specified
            if "pacing" in request.style_preferences:
                pacing = request.style_preferences["pacing"]
                if pacing == "fast":
                    style = "tiktok"
                elif pacing == "medium":
                    style = "youtube"
                elif pacing == "slow":
                    style = "cinematic"

        # 1. Generate advanced edit plan
        plan = generate_advanced_edit_plan(video_id, job.analysis_result, style=style, edit_scale=request.edit_scale)

        if request.dry_run:
            # Just return the plan
            return plan

        # 2. Convert plan to EditingTemplate and register it
        template = edit_decision_map_to_template(plan)
        template = await template_manager.create_template(template)

        # 3. Convert plan segments to timeline segments
        from app.models.schemas import VideoTimeline, TimelineSegment
        segments = []
        for seg in plan.segments:
            segments.append(TimelineSegment(
                start_time=seg.start,
                end_time=seg.end,
                source_video_id=video_id,
                effects=seg.tags,
                transition_in=seg.transition,
                transition_out=seg.transition,
            ))
        timeline = VideoTimeline(
            video_id=video_id,
            template=template,
            segments=segments,
            total_duration=job.analysis_result.duration
        )

        # 4. Create and enqueue an edit job using this template and timeline
        edit_job = await job_queue.create_editing_job(
            video_id=video_id,
            template_id=template.template_id,
            template_type=template.template_type,
            custom_settings={},
            quality_preset=template.quality_preset
        )
        edit_job.timeline = timeline
        job_queue.enqueue_job(edit_job)

        # 5. Return the job_id (frontend expects this format)
        return {
            "job_id": str(edit_job.job_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ADVANCED_EDIT ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Advanced edit failed: {e}")


@video_router.post("/{video_id}/llm_edit")
async def llm_edit(
    video_id: UUID,
    request: AdvancedEditRequest = Body(...)
):
    # Set the video_id from the URL path if not provided in body
    if request.video_id is None:
        request.video_id = video_id
    """
    Generate and render an LLM-powered intelligent edit using existing analysis results.
    """
    from app.services import get_service_manager
    from app.editor.advanced_edit import generate_llm_edit_plan
    job_queue = await get_service_manager().get_redis()
    template_manager = await get_service_manager().get_template_manager()
    try:
        job = await job_queue.get_job_by_video_id(video_id)
        if not job or not job.analysis_result:
            raise HTTPException(status_code=404, detail="Analysis results not found for this video.")

        # Extract style from style_preferences or use default
        style = "tiktok"  # default
        if request.style_preferences:
            if "energy_level" in request.style_preferences:
                energy_level = request.style_preferences["energy_level"]
                if energy_level == "high":
                    style = "tiktok"
                elif energy_level == "medium":
                    style = "youtube"
                elif energy_level == "low":
                    style = "cinematic"
            
            # Override with pacing if specified
            if "pacing" in request.style_preferences:
                pacing = request.style_preferences["pacing"]
                if pacing == "fast":
                    style = "tiktok"
                elif pacing == "medium":
                    style = "youtube"
                elif pacing == "slow":
                    style = "cinematic"

        # 1. Generate LLM-powered edit plan
        plan = generate_llm_edit_plan(
            video_id, 
            job.analysis_result, 
            style=style, 
            edit_scale=request.edit_scale,
            target_duration=request.target_duration
        )

        # Log the LLM plan as pretty-printed JSON
        import json
        # Convert UUIDs to strings for JSON serialization
        plan_dict = plan.dict() if hasattr(plan, 'dict') else plan
        # Handle UUID serialization
        def convert_uuids_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_uuids_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_uuids_for_json(item) for item in obj]
            elif hasattr(obj, 'hex'):  # UUID objects
                return str(obj)
            else:
                return obj
        
        plan_dict = convert_uuids_for_json(plan_dict)
        logger.info("[LLM_EDIT] Generated LLM editing plan:\n" + json.dumps(plan_dict, indent=2))

        # Save the plan JSON to the job metadata for traceability
        save_job_fn = getattr(job_queue, '_save_job_to_redis', None)
        if callable(save_job_fn):
            job.metadata["llm_plan_json"] = json.dumps(plan_dict)
            save_job_fn(job)

        if request.dry_run:
            # Just return the plan
            return plan

        # 2. Convert plan to EditingTemplate and register it
        template = edit_decision_map_to_template(plan)
        template = await template_manager.create_template(template)

        # 3. Convert plan segments to timeline segments
        from app.models.schemas import VideoTimeline, TimelineSegment
        segments = []
        for seg in plan.segments:
            segments.append(TimelineSegment(
                start_time=seg.start,
                end_time=seg.end,
                source_video_id=video_id,
                effects=seg.tags,
                transition_in=seg.transition,
                transition_out=seg.transition,
            ))
        timeline = VideoTimeline(
            video_id=video_id,
            template=template,
            segments=segments,
            total_duration=job.analysis_result.duration
        )

        # 4. Create and enqueue an edit job using this template and timeline
        edit_job = await job_queue.create_editing_job(
            video_id=video_id,
            template_id=template.template_id,
            template_type=template.template_type,
            custom_settings={},
            quality_preset=template.quality_preset
        )
        
        # Store LLM-specific metadata for the worker
        edit_job.metadata["llm_plan_json"] = json.dumps(plan_dict)
        edit_job.metadata["edit_scale"] = request.edit_scale
        edit_job.metadata["target_duration"] = request.target_duration
        
        edit_job.timeline = timeline
        job_queue.enqueue_job(edit_job)

        # 5. Return the job_id (frontend expects this format)
        return {
            "job_id": str(edit_job.job_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LLM_EDIT ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"LLM edit failed: {e}")


@video_router.post("/{video_id}/enhanced_llm_edit")
async def enhanced_llm_edit(
    video_id: UUID,
    request: AdvancedEditRequest = Body(...)
):
    # Set the video_id from the URL path if not provided in body
    if request.video_id is None:
        request.video_id = video_id
    """
    Generate and render an enhanced LLM-powered intelligent edit using existing analysis results.
    Uses enhanced shader layers and smart transition detection.
    """
    import traceback
    from app.services import get_service_manager
    from app.editor.enhanced_llm_editor import create_enhanced_llm_editor
    from app.models.schemas import EditStyle
    job_queue = await get_service_manager().get_redis()
    template_manager = await get_service_manager().get_template_manager()
    try:
        # For enhanced_llm_edit, we need to find the analysis job specifically
        # Get all job IDs and check each one for this video_id with analysis results
        all_job_ids = job_queue._get_all_job_ids_from_redis()
        analysis_job = None
        
        for job_id in all_job_ids:
            job = job_queue._load_job_from_redis(job_id)
            if job and job.video_id == video_id and job.analysis_result:
                analysis_job = job
                break
        
        if not analysis_job:
            raise HTTPException(status_code=404, detail="Analysis results not found for this video.")
        
        job = analysis_job

        # Convert string style to EditStyle enum
        style_str = "tiktok"  # default
        if request.style_preferences:
            if "energy_level" in request.style_preferences:
                energy_level = request.style_preferences["energy_level"]
                if energy_level == "high":
                    style_str = "tiktok"
                elif energy_level == "medium":
                    style_str = "youtube"
                elif energy_level == "low":
                    style_str = "cinematic"
            
            # Override with pacing if specified
            if "pacing" in request.style_preferences:
                pacing = request.style_preferences["pacing"]
                if pacing == "fast":
                    style_str = "tiktok"
                elif pacing == "medium":
                    style_str = "youtube"
                elif pacing == "slow":
                    style_str = "cinematic"

        # Convert to EditStyle enum
        style = EditStyle(style_str)

        # 1. Create enhanced LLM editor and generate enhanced edit plan
        enhanced_editor = create_enhanced_llm_editor("openai")
        enhanced_plan = await enhanced_editor.generate_editing_plan(
            analysis_result=job.analysis_result,
            style=style,
            target_duration=request.target_duration
        )

        # Log the enhanced LLM plan as pretty-printed JSON
        import json
        # Convert the enhanced plan to a serializable format
        plan_dict = {
            "style": enhanced_plan.style,
            "target_duration": enhanced_plan.target_duration,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "effects": seg.effects,
                    "transition_in": seg.transition_in,
                    "transition_out": seg.transition_out,
                    "source_video_id": str(seg.source_video_id),
                    "segment_order": getattr(seg, 'segment_order', None),
                    "llm_reasoning": getattr(seg, 'llm_reasoning', None),
                    "confidence_score": getattr(seg, 'confidence_score', None),
                    "segment_tags": getattr(seg, 'segment_tags', None),
                    "audio_start_time": getattr(seg, 'audio_start_time', None),
                    "audio_end_time": getattr(seg, 'audio_end_time', None),
                    "effectCustomizations": getattr(seg, 'effectCustomizations', {})
                } for seg in enhanced_plan.segments
            ],
            "transitions": enhanced_plan.transitions,
            "effects": enhanced_plan.effects,
            "reasoning": enhanced_plan.reasoning,
            "confidence": enhanced_plan.confidence,
            "transition_points_count": len(enhanced_plan.transition_points),
            "transition_segments_count": len(enhanced_plan.transition_segments),
            "smart_detection_metadata": enhanced_plan.smart_detection_metadata
        }
        
        logger.info("[ENHANCED_LLM_EDIT] Generated enhanced LLM editing plan:\n" + json.dumps(plan_dict, indent=2))

        logger.info("Saving enhanced LLM plan to job metadata...")
        # Save the enhanced plan JSON to the job metadata for traceability
        save_job_fn = getattr(job_queue, '_save_job_to_redis', None)
        if callable(save_job_fn):
            job.metadata["enhanced_llm_plan_json"] = json.dumps(plan_dict)
            save_job_fn(job)
            logger.info("✅ Enhanced LLM plan saved to job metadata")
        else:
            logger.warning("⚠️ Could not save job to Redis (save_job_fn not callable)")

        if request.dry_run:
            # Convert enhanced plan to suggestions format for frontend compatibility
            suggestions = []
            
            # Add style-based suggestions
            suggestions.append({
                "type": "style",
                "title": f"{style.value.title()} Style",
                "description": f"Apply {style.value} editing style for optimal results",
                "reasoning": enhanced_plan.reasoning,
                "confidence": enhanced_plan.confidence,
                "applied": True,
                "segment_index": -1
            })
            
            # Add segment-based suggestions from the enhanced plan
            for i, segment in enumerate(enhanced_plan.segments):  # Include ALL segments
                suggestions.append({
                    "type": "effect",
                    "title": f"{', '.join(segment.effects).replace('_', ' ').title()} ({segment.start_time:.1f}s-{segment.end_time:.1f}s)",
                    "description": f"Apply {', '.join(segment.effects)} effects from {segment.start_time:.1f}s to {segment.end_time:.1f}s",
                    "reasoning": f"Enhanced LLM analysis suggests these effects for {style.value} style",
                    "confidence": 0.85,
                    "applied": False,
                    "segment_index": i,
                    "segment_data": {
                        "start": segment.start_time,
                        "end": segment.end_time,
                        "effects": segment.effects,
                        "transition_in": segment.transition_in,
                        "transition_out": segment.transition_out,
                        "effectCustomizations": {
                            effect: {
                                "enabled": True,
                                "parameters": {"speed_factor": 2.0} if effect == "speed_up" else
                                             {"speed_factor": 0.5} if effect == "slow_motion" else
                                             {"speed_factor": getattr(segment, 'speed_factor', 1.0)} if effect == "speed" else
                                             {}
                            } for effect in segment.effects
                        }
                    }
                })
            
            # Transitions are already included in segment data, no need for separate suggestions
            
            return {
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "video_id": str(video_id),
                "style": style.value,
                "edit_scale": request.edit_scale,
                "enhanced_plan": plan_dict  # Include the full enhanced plan for debugging
            }

        # 2. Create a custom template for enhanced rendering
        from app.models.schemas import EditingTemplate, TemplateType, QualityPreset, VideoFormat
        import uuid
        enhanced_template = EditingTemplate(
            template_id=uuid.uuid4(),
            name=f"Enhanced {style.value} Edit",
            description=f"Enhanced LLM-generated {style.value} edit with shader layers",
            template_type=TemplateType.MINIMAL,  # Use MINIMAL as base, will be overridden by custom settings
            quality_preset=QualityPreset.HIGH,
            output_format=VideoFormat.MP4,
            effects=["enhanced_rendering", "shader_layers", "smart_transitions", "analysis_integrated"]
        )
        enhanced_template = await template_manager.create_template(enhanced_template)

        # 3. Create and enqueue an edit job using the enhanced template and timeline
        edit_job = await job_queue.create_editing_job(
            video_id=video_id,
            template_id=enhanced_template.template_id,
            template_type=enhanced_template.template_type,
            custom_settings={
                "enhanced_rendering": True,
                "shader_layers": True,
                "smart_transitions": True,
                "analysis_integrated": True,
                "edit_scale": request.edit_scale
            },
            quality_preset=enhanced_template.quality_preset
        )
        
        # Store enhanced LLM-specific metadata for the worker
        edit_job.metadata["enhanced_llm_plan_json"] = json.dumps(plan_dict)
        edit_job.metadata["edit_scale"] = request.edit_scale
        edit_job.metadata["target_duration"] = request.target_duration
        edit_job.metadata["enhanced_rendering"] = True
        edit_job.metadata["shader_layers"] = True
        
        # Convert enhanced plan to VideoTimeline for storage
        from app.models.schemas import VideoTimeline, TimelineSegment
        timeline_segments = []
        for seg in enhanced_plan.segments:
            # Create effectCustomizations with speed parameters for speed effects
            effectCustomizations = {}
            for effect in seg.effects:
                if effect in ["speed_up", "slow_motion", "speed"]:
                    # Add speed parameters to effectCustomizations
                    if effect == "speed_up":
                        effectCustomizations[effect] = {
                            "enabled": True,
                            "parameters": {"speed_factor": 2.0}
                        }
                    elif effect == "slow_motion":
                        effectCustomizations[effect] = {
                            "enabled": True,
                            "parameters": {"speed_factor": 0.5}
                        }
                    elif effect == "speed":
                        # Use the speed_factor from the segment if available
                        speed_factor = getattr(seg, 'speed_factor', 1.0)
                        effectCustomizations[effect] = {
                            "enabled": True,
                            "parameters": {"speed_factor": speed_factor}
                        }
                else:
                    # Regular effects
                    effectCustomizations[effect] = {
                        "enabled": True,
                        "parameters": {}
                    }
            
            timeline_segments.append(TimelineSegment(
                start_time=seg.start_time,
                end_time=seg.end_time,
                source_video_id=seg.source_video_id,
                effects=seg.effects,
                transition_in=seg.transition_in,
                transition_out=seg.transition_out,
                effectCustomizations=effectCustomizations
            ))
        
        enhanced_timeline = VideoTimeline(
            video_id=video_id,
            template=enhanced_template,
            segments=timeline_segments,
            total_duration=job.analysis_result.duration
        )
        
        # Store the enhanced timeline
        edit_job.timeline = enhanced_timeline
        
        # Enqueue the job for processing
        job_queue.enqueue_job(edit_job)

        # 5. Return the job_id (frontend expects this format)
        return {
            "job_id": str(edit_job.job_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ENHANCED_LLM_EDIT ERROR] {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Enhanced LLM edit failed: {e}")


@video_router.get("/{video_id}/stream")
async def stream_video(
    video_id: UUID,
    video_type: str = Query("processed", description="Type of video to stream: 'processed' or 'original'")
):
    """
    Stream video file directly from S3 for browser playback.
    """
    try:
        storage_client = get_storage_client()
        
        # Determine the file key based on video type
        if video_type == "original":
            file_key = f"uploads/{video_id}.mp4"
        else:
            file_key = f"processed/{video_id}_processed.mp4"
        
        # Get the video file from S3
        try:
            response = storage_client.s3_client.get_object(
                Bucket=storage_client.bucket_name,
                Key=file_key
            )
        except Exception as e:
            logger.error(f"Failed to get video file {file_key}: {e}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Get content type and size
        content_type = response.get('ContentType', 'video/mp4')
        content_length = response.get('ContentLength', 0)
        
        # Create streaming response
        return StreamingResponse(
            response['Body'],
            media_type=content_type,
            headers={
                'Content-Length': str(content_length),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600'
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stream video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {str(e)}")

@video_router.get("/{video_id}/download")
async def download_video(
    video_id: UUID,
    video_type: str = Query("processed", description="Type of video to download: 'processed' or 'original'")
):
    """
    Get download URL for processed or original video.
    """
    from app.services import get_service_manager
    storage_client = await get_service_manager().get_storage()
    job_queue = await get_service_manager().get_redis()
    template_manager = await get_service_manager().get_template_manager()
    try:
        # Get job information
        job = await job_queue.get_job_by_video_id(video_id)
        if not job:
            raise HTTPException(status_code=404, detail="Video processing job not found")
        
        if video_type == "original":
            # Download original uploaded video
            # Construct the original video path 
            original_key = f"uploads/{video_id}.mp4"
            
            # Generate download URL for original video
            download_url = await storage_client.get_video_access_url(original_key)
            
            return JSONResponse({
                "download_url": str(download_url),
                "video_id": str(video_id),
                "video_type": "original",
                "status": str(job.status)
            })
        
        else:  # processed/edited video
            # First try to get the output_url from the job if available
            if job.output_url:
                download_url = await storage_client.create_download_url(job.output_url)
                return JSONResponse({
                    "download_url": str(download_url),
                    "video_id": str(video_id),
                    "video_type": "processed",
                    "status": "completed"
                })
            # If no output_url, try to construct the edited video path with correct format
            # Get the template to determine the output format
            try:
                template = await template_manager.get_template_by_type(job.template_type)
                if template:
                    output_format = template.output_format.value
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No template found for type {job.template_type}"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get template: {str(e)}"
                )
            edited_key = f"processed/{video_id}_processed.{output_format}"
            try:
                # Check if edited video exists and generate download URL
                download_url = await storage_client.get_video_access_url(edited_key)
                return JSONResponse({
                    "download_url": str(download_url),
                    "video_id": str(video_id),
                    "video_type": "processed",
                    "status": "completed"
                })
            except Exception as e:
                # If edited video doesn't exist, check job status
                if job.status != ProcessingStatus.COMPLETED:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Video processing not completed. Current status: {job.status}. Use video_type='original' to download the source video."
                    )
                else:
                    raise HTTPException(status_code=404, detail="Processed video not found. Video may have been analyzed but not edited yet.")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate download URL for {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate download URL")


@video_router.get("/{video_id}/timeline")
async def get_video_timeline(
    video_id: UUID
):
    """
    Get the timeline for a video with segments and editing decisions.
    
    Args:
        video_id: Video identifier
        
    Returns:
        JSONResponse: Timeline data with segments and transitions
    """
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    
    try:
        # Get the latest job status to find timeline data
        job = await job_queue.get_job_by_video_id(video_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail="Video processing job not found"
            )
        
        # Extract timeline from job result
        timeline_data = job.timeline if hasattr(job, 'timeline') else None
        
        # Check if we have LLM-generated timeline data in metadata
        llm_timeline_data = None
        if hasattr(job, 'metadata') and job.metadata and 'enhanced_llm_plan_json' in job.metadata:
            try:
                import json
                llm_timeline_data = json.loads(job.metadata['enhanced_llm_plan_json'])
                logger.info(f"Found LLM-generated timeline data for video {video_id}")
            except Exception as e:
                logger.warning(f"Failed to parse LLM timeline data: {e}")
        
        # Use LLM-generated timeline if available, otherwise fall back to template timeline
        if llm_timeline_data and 'segments' in llm_timeline_data:
            timeline_data = llm_timeline_data
            logger.info(f"Using LLM-generated timeline with {len(timeline_data['segments'])} segments")
        elif not timeline_data:
            # Return a pending timeline response instead of 404
            return JSONResponse({
                "success": True,
                "video_id": str(video_id),
                "status": "pending",
                "message": "Timeline is being generated",
                "timeline": {
                    "video_id": str(video_id),
                    "style": "intelligent",
                    "target_duration": 0,
                    "segments": [],
                    "total_duration": 0
                },
                "segments": [],
                "total_duration": 0,
                "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') and job.created_at else None,
                "updated_at": datetime.utcnow().isoformat()
            })
        
        # Convert UUIDs, datetimes, and Pydantic models to strings for JSON serialization
        def convert_uuids_for_json(obj):
            if isinstance(obj, UUID):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'dict'):  # Pydantic models
                return convert_uuids_for_json(obj.dict())
            elif isinstance(obj, dict):
                return {k: convert_uuids_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_uuids_for_json(item) for item in obj]
            else:
                return obj
        
        # Handle both Pydantic models and dictionaries
        if hasattr(timeline_data, 'dict'):
            timeline_dict = timeline_data.dict()
            segments = timeline_data.segments if hasattr(timeline_data, 'segments') else []
        else:
            timeline_dict = timeline_data
            segments = timeline_data.get('segments', []) if isinstance(timeline_data, dict) else []
        
        # Convert UUIDs in timeline data
        timeline_dict = convert_uuids_for_json(timeline_dict)
        # Add video URLs to segments to avoid CORS issues
        for segment in segments:
            if isinstance(segment, dict) and 'source_video_id' in segment:
                # Only set individual video URLs if not already set to final output URL
                if 'video_url' not in segment or not segment['video_url'].startswith('https://'):
                    source_video_id = segment['source_video_id']
                    segment['video_url'] = f"/api/v1/videos/{source_video_id}/stream?video_type=original"
                    segment['stream_url'] = f"/api/v1/videos/{source_video_id}/stream?video_type=processed"
        
        segments = convert_uuids_for_json(segments)
        
        return JSONResponse({
            "success": True,
            "video_id": str(video_id),
            "timeline": timeline_dict,
            "segments": segments,
            "total_duration": timeline_data.total_duration if hasattr(timeline_data, 'total_duration') else 0,
            "created_at": job.created_at.isoformat() if hasattr(job, 'created_at') and job.created_at else None,
            "updated_at": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting timeline for video {video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get timeline: {str(e)}"
        )


@video_router.put("/{video_id}/timeline")
async def update_video_timeline(
    video_id: UUID,
    timeline_update: dict = Body(...)
):
    """
    Update the timeline for a video with manual edits.
    
    Args:
        video_id: Video identifier
        timeline_update: Updated timeline data
        
    Returns:
        JSONResponse: Updated timeline confirmation
    """
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    
    try:
        # Validate timeline update
        if "segments" not in timeline_update:
            raise HTTPException(
                status_code=400,
                detail="Timeline update must include segments"
            )
        
        # Get current job
        job = await job_queue.get_job_by_video_id(video_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail="Video processing job not found"
            )
        
        # Update timeline in job (in a real implementation, this would be saved to a database)
        # For now, we'll just log the update
        logger.info(f"Timeline updated for video {video_id}: {len(timeline_update['segments'])} segments")
        
        return JSONResponse({
            "success": True,
            "video_id": str(video_id),
            "message": "Timeline updated successfully",
            "segments_count": len(timeline_update["segments"]),
            "updated_at": datetime.utcnow().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating timeline for video {video_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update timeline: {str(e)}"
        )


@video_router.get("/", response_model=JobListResponse)
async def list_videos(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by status")
):
    """
    List all video processing jobs with pagination and filtering.
    """
    from app.services import get_service_manager
    job_queue = await get_service_manager().get_redis()
    try:
        jobs, total_count = await job_queue.list_jobs(
            page=page,
            page_size=page_size,
            status=status
        )
        
        return JobListResponse(
            jobs=jobs,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list videos: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video list")


# Template Management Routes
@template_router.get("/", response_model=TemplateListResponse)
async def list_templates(
    template_type: Optional[TemplateType] = Query(None, description="Filter by template type")
):
    """
    List available editing templates.
    """
    from app.services import get_service_manager
    template_manager = await get_service_manager().get_template_manager()
    try:
        templates = await template_manager.list_templates(template_type=template_type)
        
        return TemplateListResponse(
            templates=templates,
            total_count=len(templates)
        )
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@template_router.get("/{template_id}", response_model=EditingTemplate)
async def get_template(
    template_id: UUID
):
    """
    Get specific template details.
    """
    from app.services import get_service_manager
    template_manager = await get_service_manager().get_template_manager()
    try:
        template = await template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve template")


@template_router.post("/", response_model=EditingTemplate)
async def create_template(
    template: EditingTemplate
):
    """
    Create a new editing template.
    """
    from app.services import get_service_manager
    template_manager = await get_service_manager().get_template_manager()
    try:
        created_template = await template_manager.create_template(template)
        return created_template
        
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        raise HTTPException(status_code=500, detail="Failed to create template")


@template_router.put("/{template_id}", response_model=EditingTemplate)
async def update_template(
    template_id: UUID,
    template: EditingTemplate
):
    """
    Update an existing template.
    """
    from app.services import get_service_manager
    template_manager = await get_service_manager().get_template_manager()
    try:
        updated_template = await template_manager.update_template(template_id, template)
        if not updated_template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return updated_template
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update template")


@template_router.delete("/{template_id}", response_model=BaseResponse)
async def delete_template(
    template_id: UUID
):
    """
    Delete a template.
    """
    from app.services import get_service_manager
    template_manager = await get_service_manager().get_template_manager()
    try:
        success = await template_manager.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return BaseResponse(message="Template deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {template_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete template") 

# DEPRECATED: Use /enhanced_llm_edit with dry_run=true instead
# This endpoint is kept for backward compatibility but should not be used 

# Multi-Video Project Routes
@multi_video_router.post("/projects", response_model=MultiVideoUploadResponse)
async def create_multi_video_project(
    files: List[UploadFile] = File(...),
    project_name: str = Form(...),
    cross_video_settings: str = Form(default="{}")
):
    """Create a new multi-video project and upload all videos."""
    try:
        # Parse cross_video_settings from JSON string
        import json
        from uuid import uuid4
        from app.services import get_service_manager
        try:
            settings_dict = json.loads(cross_video_settings) if cross_video_settings else {}
        except json.JSONDecodeError:
            settings_dict = {}
        
        # Get services
        storage_client = await get_service_manager().get_storage()
        job_queue = await get_service_manager().get_redis()
        multi_video_manager = await get_multi_video_manager()
        logger.info(f"[UPLOAD] Got multi_video_manager: {multi_video_manager}")
        logger.info(f"[UPLOAD] Multi_video_manager type: {type(multi_video_manager)}")
        
        # Create multi-video project first
        logger.info(f"[UPLOAD] Creating multi-video project with name: {project_name}")
        project = await multi_video_manager.create_project(
            name=project_name,
            video_ids=[]
        )
        logger.info(f"[UPLOAD] Created project: {project.project_id}")
        logger.info(f"[UPLOAD] Project analysis_jobs: {project.analysis_jobs}")
        
        # Upload all videos
        video_ids = []
        for file in files:
            # Upload video
            video_id = uuid4()
            
            # Create temporary file for upload
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                # Read and write file content to temporary file
                file_content = await file.read()
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Upload the temporary file
                video_url = await storage_client.upload_file(
                    file_path=temp_file_path,
                    file_key=f"uploads/{video_id}.mp4",
                    content_type=file.content_type or "video/mp4"
                )
                
                # Skip analysis - new workflow uses create_robust_25_second_video.py directly
                logger.info(f"[UPLOAD] Skipping analysis for video {video_id} - using new workflow")
                
                video_ids.append(video_id)
                logger.info(f"[UPLOAD] Added video {video_id} to video_ids list")
                
                logger.info(f"✅ Uploaded video {file.filename} to project {project.project_id}")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        # After all videos are uploaded, reload the project and trigger editing directly
        project = await multi_video_manager.get_project(project.project_id)
        project.video_ids = video_ids
        await multi_video_manager.update_project(project)
        
        # Trigger editing directly since we skipped analysis
        logger.info(f"[UPLOAD] Triggering editing directly for project {project.project_id}")
        print(f"[UPLOAD] Triggering editing directly for project {project.project_id}")
        try:
            await _trigger_editing_for_project(str(project.project_id))
            logger.info(f"[UPLOAD] Successfully triggered editing for project {project.project_id}")
            print(f"[UPLOAD] Successfully triggered editing for project {project.project_id}")
        except Exception as e:
            logger.error(f"[UPLOAD] Failed to trigger editing for project {project.project_id}: {e}")
            print(f"[UPLOAD] Failed to trigger editing for project {project.project_id}: {e}")
            import traceback
            logger.error(f"[UPLOAD] Exception traceback: {traceback.format_exc()}")
            print(f"[UPLOAD] Exception traceback: {traceback.format_exc()}")
        
        # Generate upload URLs for each video (for future use)
        upload_urls = []
        expires_at = "2025-12-31T23:59:59Z"  # Set a far future expiration
        
        for video_id in video_ids:
            # Create a placeholder upload URL for each video
            upload_url = f"http://localhost:8000/api/v1/videos/{video_id}/upload"
            upload_urls.append(upload_url)
        
        return MultiVideoUploadResponse(
            project_id=project.project_id,
            video_ids=video_ids,
            upload_urls=upload_urls,
            expires_at=expires_at,
            metadata={
                "files_uploaded": len(files),
                "project_name": project.name,
                "status": "uploaded"
            }
        )
        
    except Exception as e:
        logger.error(f"Multi-video upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@multi_video_router.post("/projects/{project_id}/analyze-multiple", response_model=CrossVideoAnalysisResult)
async def analyze_multiple_videos(
    project_id: UUID,
    request: CrossVideoAnalysisRequest
):
    """Perform cross-video analysis to determine optimal combination strategy."""
    # TEMPORARILY DISABLED - Backend auto-trigger handles cross-analysis
    logger.warning(f"[DISABLED] Manual cross-analysis trigger attempted for project {project_id}. Backend auto-trigger handles this automatically.")
    raise HTTPException(
        status_code=400, 
        detail="Manual cross-analysis is disabled. The backend automatically triggers cross-analysis when all individual analysis jobs are completed."
    )

@multi_video_router.post("/projects/{project_id}/test-auto-trigger", response_model=Dict[str, Any])
async def test_auto_trigger_editing_job(project_id: UUID):
    """Test endpoint to manually trigger the auto-triggering of editing jobs."""
    try:
        logger.info(f"[TEST AUTO-TRIGGER] Testing auto-trigger for project {project_id}")
        
        # Call the LLM recommendation function directly
        await _trigger_llm_recommendation_after_cross_analysis(str(project_id))
        
        # Get the updated project to see if editing job was created
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_id)
        
        return {
            "success": True,
            "message": "Auto-trigger test completed",
            "project_id": str(project_id),
            "editing_job": str(project.editing_job) if project.editing_job else None,
            "cross_analysis_job": str(project.cross_analysis_job) if project.cross_analysis_job else None,
            "status": project.status
        }
        
    except Exception as e:
        logger.error(f"[TEST AUTO-TRIGGER] Test failed: {e}")
        import traceback
        logger.error(f"[TEST AUTO-TRIGGER] Full traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "message": f"Auto-trigger test failed: {str(e)}",
            "project_id": str(project_id),
            "error": str(e)
        }

@multi_video_router.post("/projects/{project_id}/edit-multiple", response_model=MultiVideoEditResponse)
async def edit_multiple_videos(
    project_id: UUID,
    request: MultiVideoEditRequest
):
    """Create a multi-video editing job to combine and edit multiple videos."""
    logger.info(f"=== EDIT MULTIPLE VIDEOS CALLED ===")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Request: {request}")
    
    try:
        logger.info(f"Starting edit_multiple_videos for project {project_id}")
        
        from app.services import get_service_manager
        from app.models.schemas import ProcessingStatus
        
        # Get the project
        logger.info("Getting multi-video manager...")
        try:
            multi_video_manager = await get_multi_video_manager()
            logger.info("Multi-video manager obtained successfully")
        except Exception as e:
            logger.error(f"Failed to get multi-video manager: {e}")
            raise Exception(f"Failed to get multi-video manager: {e}")
        
        logger.info("Getting project...")
        try:
            project = await multi_video_manager.get_project(project_id)
            logger.info(f"Project retrieved: {project is not None}")
            if project:
                logger.info(f"Project editing_job: {project.editing_job}")
                logger.info(f"Project cross_analysis_job: {project.cross_analysis_job}")
                logger.info(f"Project analysis_jobs: {project.analysis_jobs}")
        except Exception as e:
            logger.error(f"Failed to get project: {e}")
            raise Exception(f"Failed to get project: {e}")
        
        if not project:
            logger.error("Project not found")
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get job queue first
        logger.info("Getting job queue...")
        try:
            job_queue = await get_service_manager().get_redis()
            logger.info("Job queue obtained successfully")
        except Exception as e:
            logger.error(f"Failed to get job queue: {e}")
            raise Exception(f"Failed to get job queue: {e}")
        
        # Check if an editing job already exists for this project
        logger.info(f"Checking for existing editing job. Project editing_job: {project.editing_job}")
        
        # Check directly in Redis for the editing job
        project_key = f"multi_video_project:{project_id}"
        logger.info(f"Checking Redis key: {project_key}")
        try:
            project_data = job_queue.redis_conn.get(project_key)
            logger.info(f"Redis project_data exists: {project_data is not None}")
            if project_data:
                import json
                data = json.loads(project_data.decode('utf-8'))
                redis_editing_job = data.get('editing_job')
                logger.info(f"Redis editing_job: {redis_editing_job}")
                logger.info(f"Redis editing_job type: {type(redis_editing_job)}")
                
                # Use the Redis editing_job if it exists, regardless of the loaded project
                if redis_editing_job:
                    logger.info(f"Editing job already exists for project {project_id}: {redis_editing_job}")
                    return MultiVideoEditResponse(
                        project_id=project_id,
                        editing_job_id=redis_editing_job,
                        status="already_exists",
                        estimated_duration=None
                    )
                else:
                    logger.info(f"No editing job found in Redis data")
            else:
                logger.info(f"No project data found in Redis")
        except Exception as e:
            logger.error(f"Failed to check Redis directly: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"No existing editing job found for project {project_id}")
        
        # Check if cross-analysis job exists
        logger.info(f"Project cross_analysis_job: {project.cross_analysis_job}")
        if not project.cross_analysis_job:
            logger.error("Cross-analysis job not set")
            raise HTTPException(
                status_code=400, 
                detail="Cross-video analysis must be completed first"
            )
        
        logger.info("Loading cross-analysis job...")
        logger.info(f"Attempting to load job {project.cross_analysis_job} from Redis...")
        try:
            cross_job = job_queue._load_job_from_redis(project.cross_analysis_job)
            logger.info(f"Cross-analysis job loaded: {cross_job is not None}")
            if cross_job:
                logger.info(f"Cross-analysis job status: {cross_job.status}")
                logger.info(f"Cross-analysis job has analysis_result: {hasattr(cross_job, 'analysis_result')}")
            else:
                logger.error("Cross-analysis job is None after loading")
        except Exception as e:
            logger.error(f"Failed to load cross-analysis job: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to load cross-analysis job: {e}")
        
        if not cross_job:
            logger.error(f"Cross-analysis job {project.cross_analysis_job} not found")
            # Try to list all keys in Redis to see what's available
            try:
                all_keys = job_queue.redis_conn.keys("*")
                logger.info(f"All Redis keys: {all_keys}")
            except Exception as e:
                logger.error(f"Failed to list Redis keys: {e}")
            
            # For now, let's proceed with the editing even if the job loading fails
            # since we know the job exists in Redis
            logger.info("Proceeding with editing despite job loading failure")
        else:
            logger.info("Cross-analysis job loaded successfully")
        
        # Check if cross-analysis job has completed
        if cross_job and cross_job.status != ProcessingStatus.COMPLETED:
            logger.info(f"Cross-analysis job {project.cross_analysis_job} is still in progress (status: {cross_job.status})")
            raise HTTPException(
                status_code=400, 
                detail=f"Cross-analysis job is still in progress (status: {cross_job.status}). Please wait for it to complete before starting editing."
            )
        
        logger.info("Cross-analysis job completed successfully")
        logger.info("Cross-analysis job found, checking for existing editing job...")
        
        # Check if an editing job already exists for this project
        project_key = f"multi_video_project:{project_id}"
        existing_editing_job = None
        try:
            project_data = job_queue.redis_conn.get(project_key)
            if project_data:
                import json
                data = json.loads(project_data.decode('utf-8'))
                existing_editing_job = data.get('editing_job')
                logger.info(f"Found existing editing job: {existing_editing_job}")
        except Exception as e:
            logger.error(f"Failed to check for existing editing job: {e}")
        
        if existing_editing_job:
            logger.info(f"Editing job already exists for project {project_id}: {existing_editing_job}")
            return MultiVideoEditResponse(
                project_id=project_id,
                editing_job_id=existing_editing_job,
                status="already_exists",
                estimated_duration=None
            )
        
        # Create new editing job for multi-video editing (bypass individual video analysis check)
        from app.models.schemas import TemplateType, QualityPreset, ProcessingJob, ProcessingStatus
        from uuid import uuid4
        
        # Get the analysis results from the individual analysis jobs
        analysis_results = []
        analysis_job_ids = []
        
        # Find analysis jobs for each video in the project
        all_job_ids = job_queue._get_all_job_ids_from_redis()
        for job_id in all_job_ids:
            job = job_queue._load_job_from_redis(job_id)
            if job and job.video_id in project.video_ids and job.analysis_result:
                analysis_results.append(job.analysis_result)
                analysis_job_ids.append(job.job_id)
                logger.info(f"Found analysis result for video {job.video_id} in job {job.job_id}")
        
        if not analysis_results:
            logger.error("No analysis results found for any videos in the project")
            raise HTTPException(
                status_code=400,
                detail="No analysis results found. Please ensure all video analysis is completed before editing."
            )
        
        # Use the first analysis result as the primary one for the editing job
        primary_analysis_result = analysis_results[0]
        logger.info(f"Using analysis result from job {analysis_job_ids[0] if analysis_job_ids else 'none'} as primary")
        
        # Get cross-video analysis results
        logger.info("Getting cross-video analysis results...")
        cross_analysis_job = job_queue._load_job_from_redis(project.cross_analysis_job)
        cross_analysis_result = None
        if cross_analysis_job:
            # Check for cross-analysis result in metadata first (new format)
            if cross_analysis_job.metadata.get('cross_analysis_result'):
                cross_analysis_result = cross_analysis_job.metadata.get('cross_analysis_result')
                logger.info("✅ Found cross-video analysis result in job metadata (cross_analysis_result)")
            elif cross_analysis_job.analysis_result and hasattr(cross_analysis_job.analysis_result, 'similarity_matrix'):
                # This is already a CrossVideoAnalysisResult
                cross_analysis_result = cross_analysis_job.analysis_result
                logger.info("✅ Found cross-video analysis result in job analysis_result")
            elif cross_analysis_job.metadata.get('cross_video_analysis'):
                # Cross-video analysis is stored in job metadata (old format)
                cross_analysis_result = cross_analysis_job.metadata.get('cross_video_analysis')
                logger.info("✅ Found cross-video analysis result in job metadata (cross_video_analysis)")
            else:
                logger.warning("⚠️ Cross-analysis job found but no cross-video analysis data")
        else:
            logger.warning("⚠️ No cross-video analysis result found")
        
        # Generate LLM recommendation for multi-video editing
        logger.info("Generating LLM recommendation for multi-video editing...")
        try:
            from app.editor.enhanced_llm_editor import create_enhanced_llm_editor
            from app.models.schemas import EditStyle
            
            logger.info("Creating enhanced LLM editor...")
            # Create enhanced LLM editor
            enhanced_editor = create_enhanced_llm_editor("openai")
            logger.info("✅ Enhanced LLM editor created successfully")
            
            # Determine style based on request settings
            logger.info("Determining editing style...")
            style_str = "tiktok"  # default
            if request.style_preferences:
                if "energy_level" in request.style_preferences:
                    energy_level = request.style_preferences["energy_level"]
                    if energy_level == "high":
                        style_str = "tiktok"
                    elif energy_level == "medium":
                        style_str = "youtube"
                    elif energy_level == "low":
                        style_str = "cinematic"
                
                # Override with pacing if specified
                if "pacing" in request.style_preferences:
                    pacing = request.style_preferences["pacing"]
                    if pacing == "fast":
                        style_str = "tiktok"
                    elif pacing == "medium":
                        style_str = "youtube"
                    elif pacing == "slow":
                        style_str = "cinematic"
            
            style = EditStyle(style_str)
            logger.info(f"✅ Editing style determined: {style_str}")
            
            # Create a combined analysis context for multi-video editing
            logger.info("Creating combined analysis context...")
            # This includes individual video analysis + cross-video analysis
            combined_analysis_context = {
                "primary_analysis": primary_analysis_result,
                "all_analysis_results": analysis_results,
                "cross_video_analysis": cross_analysis_result,
                "video_ids": [str(vid) for vid in project.video_ids],
                "analysis_job_ids": [str(job_id) for job_id in analysis_job_ids],
                "cross_analysis_job_id": str(project.cross_analysis_job),
                "edit_settings": request.dict()
            }
            logger.info(f"✅ Combined analysis context created with {len(analysis_results)} analysis results")
            
            # Log detailed breakdown of what we're sending to LLM
            logger.info("🎯 [LLM INPUT] Detailed breakdown of analysis data being sent to LLM:")
            logger.info("=" * 80)
            
            # Log primary analysis
            logger.info("📊 PRIMARY ANALYSIS (Video 1):")
            if isinstance(primary_analysis_result, dict):
                logger.info(f"   Keys: {list(primary_analysis_result.keys())}")
                if 'motion_analysis' in primary_analysis_result:
                    motion = primary_analysis_result['motion_analysis']
                    logger.info(f"   Motion Analysis: {motion.get('average_motion_intensity', 'N/A')} intensity")
                if 'audio_analysis' in primary_analysis_result:
                    audio = primary_analysis_result['audio_analysis']
                    logger.info(f"   Audio Analysis: {len(audio.get('beats', []))} beats, {audio.get('tempo', 'N/A')} BPM")
                if 'metadata' in primary_analysis_result:
                    meta = primary_analysis_result['metadata']
                    logger.info(f"   Metadata: {meta.get('duration', 'N/A')}s, {meta.get('fps', 'N/A')} FPS")
            else:
                logger.info(f"   Type: {type(primary_analysis_result)}")
            
            # Log all analysis results
            logger.info(f"📊 ALL ANALYSIS RESULTS ({len(analysis_results)} videos):")
            for i, analysis in enumerate(analysis_results):
                logger.info(f"   Video {i+1}:")
                if isinstance(analysis, dict):
                    logger.info(f"     Keys: {list(analysis.keys())}")
                    if 'motion_analysis' in analysis:
                        motion = analysis['motion_analysis']
                        logger.info(f"     Motion: {motion.get('average_motion_intensity', 'N/A')} intensity")
                    if 'audio_analysis' in analysis:
                        audio = analysis['audio_analysis']
                        logger.info(f"     Audio: {len(audio.get('beats', []))} beats, {audio.get('tempo', 'N/A')} BPM")
                    if 'metadata' in analysis:
                        meta = analysis['metadata']
                        logger.info(f"     Metadata: {meta.get('duration', 'N/A')}s, {meta.get('fps', 'N/A')} FPS")
                else:
                    logger.info(f"     Type: {type(analysis)}")
            
            # Log cross-analysis
            logger.info("🔗 CROSS-VIDEO ANALYSIS:")
            if hasattr(cross_analysis_result, 'metadata') and isinstance(cross_analysis_result.metadata, dict):
                cross_meta = cross_analysis_result.metadata
                logger.info(f"   Analysis Method: {cross_meta.get('analysis_method', 'N/A')}")
                if 'analysis_summary' in cross_meta:
                    summary = cross_meta['analysis_summary']
                    logger.info(f"   Summary: {summary.get('total_videos', 'N/A')} videos, {summary.get('total_segments', 'N/A')} segments")
                    logger.info(f"   Average Similarity: {summary.get('average_similarity', 'N/A')}")
                if 'individual_analyses' in cross_meta:
                    logger.info(f"   Individual Analyses: {len(cross_meta['individual_analyses'])} videos included")
            else:
                logger.info(f"   Type: {type(cross_analysis_result)}")
            
            # Log similarity matrix
            if hasattr(cross_analysis_result, 'similarity_matrix'):
                logger.info("📈 SIMILARITY MATRIX:")
                for vid1, similarities in cross_analysis_result.similarity_matrix.items():
                    for vid2, similarity in similarities.items():
                        logger.info(f"   {vid1[:8]} -> {vid2[:8]}: {similarity:.3f}")
            
            # Log chunking points
            if hasattr(cross_analysis_result, 'cross_video_segments'):
                logger.info(f"🎬 CHUNKING POINTS: {len(cross_analysis_result.cross_video_segments)} total")
                for i, point in enumerate(cross_analysis_result.cross_video_segments[:5]):  # Show first 5
                    logger.info(f"   Point {i+1}: {point.get('chunk_type', 'N/A')} at {point.get('start_time', 0):.1f}s")
                if len(cross_analysis_result.cross_video_segments) > 5:
                    logger.info(f"   ... and {len(cross_analysis_result.cross_video_segments) - 5} more")
            
            logger.info("=" * 80)
            
            logger.info("Generating enhanced editing plan with cross-video context...")
            # Generate enhanced editing plan with cross-video context
            enhanced_plan = await enhanced_editor.generate_editing_plan(
                analysis_result=primary_analysis_result,
                style=style,
                target_duration=request.target_duration,
                multi_video_context=combined_analysis_context  # Pass cross-video context
            )
            logger.info(f"✅ Enhanced editing plan generated: {len(enhanced_plan.segments)} segments")
            
            # Log the LLM response in detail
            logger.info("🎯 [LLM RESPONSE] Detailed breakdown of LLM recommendation:")
            logger.info("=" * 80)
            
            # Convert the enhanced plan to a serializable format
            import json
            plan_dict = {
                "style": enhanced_plan.style,
                "target_duration": enhanced_plan.target_duration,
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "effects": seg.effects,
                        "transition_in": seg.transition_in,
                        "transition_out": seg.transition_out,
                        "source_video_id": str(seg.source_video_id)
                    } for seg in enhanced_plan.segments
                ],
                "transitions": enhanced_plan.transitions,
                "effects": enhanced_plan.effects,
                "reasoning": enhanced_plan.reasoning,
                "confidence": enhanced_plan.confidence,
                "transition_points_count": len(enhanced_plan.transition_points),
                "transition_segments_count": len(enhanced_plan.transition_segments),
                "smart_detection_metadata": enhanced_plan.smart_detection_metadata
            }
            
            # Log the complete LLM response
            logger.info("📋 COMPLETE LLM RESPONSE:")
            logger.info(json.dumps(plan_dict, indent=2, default=str))
            
            logger.info("🎯 [MULTI-VIDEO LLM] LLM Recommendation Summary:")
            logger.info(f"   Style: {plan_dict.get('style', 'unknown')}")
            logger.info(f"   Target Duration: {plan_dict.get('target_duration', 'unknown')}")
            logger.info(f"   Segments: {len(plan_dict.get('segments', []))}")
            logger.info(f"   Effects: {plan_dict.get('effects', [])}")
            logger.info(f"   Transitions: {plan_dict.get('transitions', [])}")
            logger.info(f"   Confidence: {plan_dict.get('confidence', 'unknown')}")
            logger.info(f"   Reasoning: {plan_dict.get('reasoning', 'unknown')[:200]}...")
            
            # Log each segment in detail
            logger.info("🎬 SEGMENT BREAKDOWN:")
            for i, segment in enumerate(plan_dict.get('segments', [])):
                logger.info(f"   Segment {i+1}: {segment.get('start_time', 0):.1f}s - {segment.get('end_time', 0):.1f}s")
                logger.info(f"     Effects: {segment.get('effects', [])}")
                logger.info(f"     Transitions: {segment.get('transition_in', 'none')} -> {segment.get('transition_out', 'none')}")
                logger.info(f"     Source Video: {segment.get('source_video_id', 'unknown')}")
            
            logger.info("=" * 80)
            
            # Create editing job with LLM recommendation
            editing_job = ProcessingJob(
                job_id=uuid4(),
                video_id=project.video_ids[0],  # Use first video as primary
                status=ProcessingStatus.PENDING,
                template_type=TemplateType.BEAT_MATCH,
                quality_preset=QualityPreset.HIGH,
                analysis_result=primary_analysis_result  # Attach the analysis result
            )
            
            # Update job metadata for multi-video editing with LLM plan
            editing_job.metadata.update({
                "job_type": "multi_video_editing",
                "project_id": str(project_id),
                "video_ids": [str(vid) for vid in project.video_ids],
                "analysis_job_ids": [str(job_id) for job_id in analysis_job_ids],
                "cross_analysis_job_id": str(project.cross_analysis_job),
                "edit_settings": request.dict(),
                "all_analysis_results": [str(result) for result in analysis_results],  # Store all results
                "enhanced_llm_plan_json": json.dumps(plan_dict)  # Add the LLM recommendation
            })
            
            logger.info("✅ LLM recommendation generated and attached to editing job")
            
            # Save the updated job
            job_queue._save_job_to_redis(editing_job)
            logger.info(f"Multi-video editing job created: {editing_job.job_id}")
            
            # Enqueue the job for processing
            job_queue.enqueue_job(editing_job)
            logger.info(f"Multi-video editing job enqueued: {editing_job.job_id}")
            
            editing_job_id = editing_job.job_id
            
            # Set the editing job in the project
            logger.info("Setting editing job in project...")
            try:
                await multi_video_manager.set_editing_job(
                    project_id=project_id,
                    editing_job_id=editing_job_id
                )
                logger.info("Editing job set successfully")
            except Exception as e:
                logger.error(f"Failed to set editing job: {e}")
                raise Exception(f"Failed to set editing job: {e}")
            
            logger.info("Returning success response")
            return MultiVideoEditResponse(
                project_id=project_id,
                editing_job_id=editing_job_id,
                status="created",
                estimated_duration=None
            )
        
        except Exception as e:
            logger.error(f"Failed to generate LLM recommendation: {e}")
            import traceback
            logger.error(f"LLM Error Traceback: {traceback.format_exc()}")
            raise Exception(f"LLM recommendation generation failed: {e}")
            await multi_video_manager.set_editing_job(
                project_id=project_id,
                editing_job_id=editing_job_id
            )
            logger.info("Editing job set successfully")
        except Exception as e:
            logger.error(f"Failed to set editing job: {e}")
            raise Exception(f"Failed to set editing job: {e}")
        
        logger.info("Returning success response")
        return MultiVideoEditResponse(
            project_id=project_id,
            editing_job_id=editing_job_id,
            status="created",
            estimated_duration=None
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Multi-video editing failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we have a meaningful error message
        if not e:
            error_msg = "Unknown error occurred during multi-video editing"
        elif not str(e):
            error_msg = "Empty error message - check logs for details"
        else:
            error_msg = str(e)
        
        logger.error(f"Final error message: '{error_msg}'")
        raise HTTPException(status_code=500, detail=f"Editing failed: {error_msg}")

@multi_video_router.get("/projects/{project_id}/status", response_model=MultiVideoProjectStatus)
async def get_multi_video_project_status(project_id: UUID):
    """Get the current status of a multi-video project."""
    try:
        from app.services import get_service_manager
        
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project_status(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Count completed analysis jobs
        job_queue = await get_service_manager().get_redis()
        analysis_completed = 0
        
        # For multi-video projects, check if we have individual analysis jobs or cross-analysis
        if project.analysis_jobs:
            # Individual analysis jobs exist
            for job_id in project.analysis_jobs:
                job = job_queue._load_job_from_redis(job_id)
                if job and job.status == ProcessingStatus.COMPLETED:
                    analysis_completed += 1
        elif project.cross_analysis_job:
            # Multi-video project with cross-analysis only
            cross_job = job_queue._load_job_from_redis(project.cross_analysis_job)
            if cross_job and cross_job.status == ProcessingStatus.COMPLETED:
                analysis_completed = len(project.video_ids)  # Count all videos as analyzed
        else:
            # No analysis jobs found - check if editing is completed
            # If editing is completed, assume analysis was skipped and mark as completed
            if project.editing_job:
                edit_job = job_queue._load_job_from_redis(project.editing_job)
                if edit_job and edit_job.status == ProcessingStatus.COMPLETED:
                    analysis_completed = len(project.video_ids)  # Mark all videos as analyzed
                else:
                    analysis_completed = 0
            else:
                analysis_completed = 0
        
        # Check cross-analysis status
        cross_analysis_completed = False
        if project.cross_analysis_job:
            cross_job = job_queue._load_job_from_redis(project.cross_analysis_job)
            logger.info(f"[STATUS] Cross-analysis job {project.cross_analysis_job}: loaded={cross_job is not None}, status={cross_job.status if cross_job else 'None'}")
            cross_analysis_completed = bool(cross_job and cross_job.status == ProcessingStatus.COMPLETED)
            logger.info(f"[STATUS] Cross-analysis completed: {cross_analysis_completed}")
        else:
            # No cross-analysis job - check if editing is completed
            # If editing is completed, assume cross-analysis was skipped and mark as completed
            if project.editing_job:
                edit_job = job_queue._load_job_from_redis(project.editing_job)
                if edit_job and edit_job.status == ProcessingStatus.COMPLETED:
                    cross_analysis_completed = True  # Mark cross-analysis as completed
                    logger.info(f"[STATUS] No cross-analysis job, but editing completed - marking cross-analysis as completed")
        
        # Check editing status
        editing_completed = False
        output_video_url = None
        if project.editing_job:
            edit_job = job_queue._load_job_from_redis(project.editing_job)
            editing_completed = bool(edit_job and edit_job.status == ProcessingStatus.COMPLETED)
            if editing_completed and edit_job and edit_job.output_url:
                output_video_url = str(edit_job.output_url)
        
        # Calculate progress
        # For multi-video: analysis (30%) + cross-analysis (30%) + editing (40%)
        if analysis_completed == len(project.video_ids) and cross_analysis_completed and editing_completed:
            progress = 100.0
        elif analysis_completed == len(project.video_ids) and cross_analysis_completed:
            progress = 70.0
        elif analysis_completed == len(project.video_ids):
            progress = 40.0
        else:
            # During analysis phase, show progress based on completed analysis jobs
            analysis_progress = (analysis_completed / len(project.video_ids)) * 40 if len(project.video_ids) > 0 else 0
            progress = analysis_progress
        
        # Get metadata including LLM plan if editing is completed
        metadata = project.metadata.copy() if project.metadata else {}
        
        # If editing is completed, try to get the LLM plan from the editing job
        if editing_completed and project.editing_job:
            edit_job = job_queue._load_job_from_redis(project.editing_job)
            if edit_job and edit_job.metadata:
                # Include the enhanced LLM plan in metadata
                if 'enhanced_llm_plan_json' in edit_job.metadata:
                    try:
                        import json
                        llm_plan = json.loads(edit_job.metadata['enhanced_llm_plan_json'])
                        metadata['enhanced_llm_plan_json'] = edit_job.metadata['enhanced_llm_plan_json']
                        metadata['llm_plan'] = llm_plan
                    except Exception as e:
                        logger.warning(f"Failed to parse LLM plan: {e}")
        
        # Add cache-busting headers to response
        response = MultiVideoProjectStatus(
            project_id=project_id,
            status=project.status,
            video_ids=project.video_ids,
            progress=progress,
            analysis_completed=analysis_completed,
            cross_analysis_job=project.cross_analysis_job,
            cross_analysis_completed=cross_analysis_completed,
            editing_completed=editing_completed,
            output_video_url=output_video_url,
            error=None,
            metadata=metadata
        )
        
        # Convert UUID objects to strings for JSON serialization
        response_dict = response.model_dump()
        response_dict['project_id'] = str(response_dict['project_id'])
        response_dict['video_ids'] = [str(vid) for vid in response_dict['video_ids']]
        if response_dict.get('cross_analysis_job'):
            response_dict['cross_analysis_job'] = str(response_dict['cross_analysis_job'])
        
        # Convert Url objects to strings for JSON serialization
        if response_dict.get('output_video_url'):
            response_dict['output_video_url'] = str(response_dict['output_video_url'])
        
        # Return with cache-busting headers
        return JSONResponse(
            content=response_dict,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise

@multi_video_router.get("/projects/{project_id}/timeline")
async def get_multi_video_project_timeline(project_id: UUID):
    """Get the timeline (LLM plan) for a multi-video project."""
    try:
        # Add cache-busting headers
        response_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        from app.services import get_service_manager
        
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project_status(project_id)
        
        if not project:
            return JSONResponse(
                status_code=404,
                content={"error": "Project not found"},
                headers=response_headers
            )
        
        # Get the editing job to extract timeline data
        job_queue = await get_service_manager().get_redis()
        timeline_data = None
        
        # Always create timeline using the final output video
        if project.status == ProcessingStatus.COMPLETED and project.editing_job:
            # Get the output video URL from the edit job
            edit_job = job_queue._load_job_from_redis(project.editing_job)
            output_video_url = None
            if edit_job and hasattr(edit_job, 'output_url') and edit_job.output_url:
                output_video_url = str(edit_job.output_url)
                logger.info(f"[TIMELINE] Using output video URL: {output_video_url}")
            else:
                logger.warning(f"[TIMELINE] No output video URL found, using fallback")
                # Fallback: construct URL from project ID
                output_video_url = f"https://my-video-editing-app-2025.s3.amazonaws.com/outputs/multi_video_output_{project_id}.mp4"
            
            # Create timeline based on the video count using the final output video
            timeline_segments = []
            segment_duration = 8.0  # 8 seconds per video
            
            for i, video_id in enumerate(project.video_ids):
                # Always use the final output video URL for all segments since it's a combined video
                video_url = output_video_url
                stream_url = output_video_url
                
                logger.info(f"[TIMELINE] Segment {i}: video_url={video_url}, stream_url={stream_url}")
                
                timeline_segments.append({
                    "start_time": i * segment_duration,
                    "end_time": (i + 1) * segment_duration,
                    "source_video_id": str(video_id),
                    "effects": ["basic_edit"],
                    "transitions": ["fade_in", "fade_out"],
                    "reasoning": "Basic intelligent editing applied",
                    "video_url": video_url,
                    "stream_url": stream_url
                })
            
            timeline_data = {
                "project_id": str(project_id),
                "style": "intelligent",
                "target_duration": len(project.video_ids) * segment_duration,
                "segments": timeline_segments,
                "total_duration": len(project.video_ids) * segment_duration
            }
        else:
            # Return a pending timeline response instead of 404
            return JSONResponse(
                status_code=200,
                content={
                    "project_id": str(project_id),
                    "status": "pending",
                    "message": "Timeline is being generated",
                    "timeline": {
                        "project_id": str(project_id),
                        "style": "intelligent",
                        "target_duration": 0,
                        "segments": [],
                        "total_duration": 0
                    },
                    "segments": [],
                    "total_duration": 0,
                    "style": "intelligent",
                    "segment_ordering": {
                        "optimal_sequence": [],
                        "original_video_order": [],
                        "has_rearrangement": False,
                        "rearrangement_reasoning": "Timeline generation in progress"
                    },
                    "llm_suggestions": []
                },
                headers=response_headers
            )
        
        # Extract multi-video metadata if available
        multi_video_metadata = {}
        optimal_sequence = []
        original_video_order = len(project.video_ids)
        
        # Try to get metadata from the LLM plan if available
        if project.editing_job:
            edit_job = job_queue._load_job_from_redis(project.editing_job)
            if edit_job and edit_job.metadata and 'enhanced_llm_plan_json' in edit_job.metadata:
                try:
                    import json
                    llm_plan = json.loads(edit_job.metadata['enhanced_llm_plan_json'])
                    multi_video_metadata = llm_plan.get("multi_video_metadata", {})
                    optimal_sequence = multi_video_metadata.get("optimal_sequence", [])
                    original_video_order = multi_video_metadata.get("video_count", len(project.video_ids))
                except Exception as e:
                    logger.warning(f"Failed to parse LLM plan metadata: {e}")
        
        # Add segment ordering information
        segment_ordering = {
            "optimal_sequence": optimal_sequence,
            "original_video_order": list(range(original_video_order)),
            "has_rearrangement": len(optimal_sequence) > 0 and optimal_sequence != list(range(original_video_order)),
            "rearrangement_reasoning": multi_video_metadata.get("rearrangement_reasoning", "LLM determined optimal flow")
        }
        
        # Generate LLM suggestions from segments
        llm_suggestions = []
        segments = timeline_data.get("segments", [])
        
        for i, segment in enumerate(segments):
            # Create suggestions based on segment effects and reasoning
            if segment.get("effects"):
                for effect in segment["effects"]:
                    suggestion = {
                        "type": "effect",
                        "title": f"Apply {effect.replace('_', ' ').title()} Effect",
                        "description": f"AI recommends applying {effect} effect to enhance this segment",
                        "reasoning": segment.get("reasoning", f"Selected {effect} effect for optimal visual impact"),
                        "confidence": 0.85,
                        "applied": True,
                        "segment_index": i,
                        "segment_data": {
                            "start": segment.get("start_time", 0),
                            "end": segment.get("end_time", 0),
                            "effect": effect,
                            "intensity": 1.0
                        }
                    }
                    llm_suggestions.append(suggestion)
            

        
        # Add video URLs to segments to avoid CORS issues
        segments = timeline_data.get("segments", [])
        for segment in segments:
            if isinstance(segment, dict) and 'source_video_id' in segment:
                # Only set individual video URLs if not already set to final output URL
                if 'video_url' not in segment or not segment['video_url'].startswith('https://'):
                    source_video_id = segment['source_video_id']
                    segment['video_url'] = f"/api/v1/videos/{source_video_id}/stream?video_type=original"
                    segment['stream_url'] = f"/api/v1/videos/{source_video_id}/stream?video_type=processed"
        
        return JSONResponse(
            status_code=200,
            content={
                "project_id": str(project_id),
                "timeline": timeline_data,
                "segments": segments,
                "total_duration": timeline_data.get("total_duration", 0),
                "style": timeline_data.get("style", "cinematic"),
                "segment_ordering": segment_ordering,
                "llm_suggestions": llm_suggestions
            },
            headers=response_headers
        )
    except Exception as e:
        logger.error(f"Error getting multi-video project timeline: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

# Helper functions
async def _perform_cross_video_analysis(
    project_id: UUID,
    video_ids: List[UUID],
    analysis_results: List[Any],
    settings: Dict[str, Any]
) -> CrossVideoAnalysisResult:
    """Perform cross-video analysis to determine optimal combination."""
    logger.info(f"[CROSS ANALYSIS] Starting real cross-video analysis for project {project_id}")
    logger.info(f"[CROSS ANALYSIS] Video IDs: {video_ids}")
    logger.info(f"[CROSS ANALYSIS] Analysis results count: {len(analysis_results)}")
    
    if len(analysis_results) < 2:
        logger.error(f"[CROSS ANALYSIS] Need at least 2 analysis results, got {len(analysis_results)}")
        raise ValueError(f"Need at least 2 analysis results, got {len(analysis_results)}")
    
    # Extract individual video analysis results
    video_1_analysis = analysis_results[0]
    video_2_analysis = analysis_results[1]
    
    logger.info(f"[CROSS ANALYSIS] Video 1 analysis keys: {list(video_1_analysis.keys()) if isinstance(video_1_analysis, dict) else 'Not a dict'}")
    logger.info(f"[CROSS ANALYSIS] Video 2 analysis keys: {list(video_2_analysis.keys()) if isinstance(video_2_analysis, dict) else 'Not a dict'}")
    
    # Calculate real similarities based on analysis results
    video_similarities = {}
    
    # Initialize similarity matrix
    for video_id in video_ids:
        video_similarities[str(video_id)] = {}
        for other_video_id in video_ids:
            if str(video_id) != str(other_video_id):
                video_similarities[str(video_id)][str(other_video_id)] = 0.0
    
    # Calculate content similarity
    if isinstance(video_1_analysis, dict) and isinstance(video_2_analysis, dict):
        # Compare motion patterns
        motion_similarity = 0.0
        if 'motion_analysis' in video_1_analysis and 'motion_analysis' in video_2_analysis:
            motion_1 = video_1_analysis.get('motion_analysis', {})
            motion_2 = video_2_analysis.get('motion_analysis', {})
            
            # Compare motion intensity
            intensity_1 = motion_1.get('average_motion_intensity', 0.0)
            intensity_2 = motion_2.get('average_motion_intensity', 0.0)
            intensity_diff = abs(intensity_1 - intensity_2)
            motion_similarity = max(0.0, 1.0 - (intensity_diff / max(intensity_1, intensity_2, 1.0)))
        
        # Compare audio characteristics
        audio_similarity = 0.0
        if 'audio_analysis' in video_1_analysis and 'audio_analysis' in video_2_analysis:
            audio_1 = video_1_analysis.get('audio_analysis', {})
            audio_2 = video_2_analysis.get('audio_analysis', {})
            
            # Compare beat patterns
            beats_1 = audio_1.get('beats', [])
            beats_2 = audio_2.get('beats', [])
            if beats_1 and beats_2:
                # Calculate beat pattern similarity
                avg_beat_interval_1 = sum(beats_1) / len(beats_1) if beats_1 else 0
                avg_beat_interval_2 = sum(beats_2) / len(beats_2) if beats_2 else 0
                beat_diff = abs(avg_beat_interval_1 - avg_beat_interval_2)
                audio_similarity = max(0.0, 1.0 - (beat_diff / max(avg_beat_interval_1, avg_beat_interval_2, 1.0)))
        
        # Compare metadata
        metadata_similarity = 0.0
        if 'metadata' in video_1_analysis and 'metadata' in video_2_analysis:
            meta_1 = video_1_analysis.get('metadata', {})
            meta_2 = video_2_analysis.get('metadata', {})
            
            # Compare duration
            duration_1 = meta_1.get('duration', 0.0)
            duration_2 = meta_2.get('duration', 0.0)
            if duration_1 > 0 and duration_2 > 0:
                duration_diff = abs(duration_1 - duration_2)
                metadata_similarity = max(0.0, 1.0 - (duration_diff / max(duration_1, duration_2)))
        
        # Calculate overall similarity
        overall_similarity = (motion_similarity + audio_similarity + metadata_similarity) / 3.0
        
        # Set similarity for both directions
        video_similarities[str(video_ids[0])][str(video_ids[1])] = overall_similarity
        video_similarities[str(video_ids[1])][str(video_ids[0])] = overall_similarity
        
        logger.info(f"[CROSS ANALYSIS] Calculated similarities:")
        logger.info(f"[CROSS ANALYSIS]   Motion similarity: {motion_similarity:.3f}")
        logger.info(f"[CROSS ANALYSIS]   Audio similarity: {audio_similarity:.3f}")
        logger.info(f"[CROSS ANALYSIS]   Metadata similarity: {metadata_similarity:.3f}")
        logger.info(f"[CROSS ANALYSIS]   Overall similarity: {overall_similarity:.3f}")
    
    # Determine optimal sequence based on similarity
    # For now, keep original order but this could be optimized
    optimal_sequence = video_ids.copy()
    
    # Generate chunking points based on analysis
    chunking_points = []
    
    # Add chunking points based on individual video analysis
    for i, (video_id, analysis_result) in enumerate(zip(video_ids, analysis_results)):
        if isinstance(analysis_result, dict):
            # Add scene transitions if available
            scenes = analysis_result.get('scene_analysis', {}).get('scenes', [])
            for scene in scenes:
                chunking_points.append({
                    "video_id": str(video_id),
                    "position": i,
                    "chunk_type": "scene",
                    "start_time": scene.get('start_time', 0.0),
                    "end_time": scene.get('end_time', 0.0),
                    "confidence": scene.get('confidence', 0.8)
                })
            
            # Add beat points if available
            audio_analysis = analysis_result.get('audio_analysis', {})
            beats = audio_analysis.get('beats', [])
            for beat_time in beats:
                chunking_points.append({
                    "video_id": str(video_id),
                    "position": i,
                    "chunk_type": "beat",
                    "start_time": beat_time,
                    "end_time": beat_time + 0.1,  # 100ms beat window
                    "confidence": 0.9
                })
    
    # Create a real cross-analysis job ID
    from uuid import uuid4
    cross_analysis_job_id = uuid4()
    
    # Create comprehensive cross-analysis result
    cross_analysis_result = CrossVideoAnalysisResult(
        project_id=project_id,
        analysis_job_id=cross_analysis_job_id,  # Use cross_analysis_job_id instead of undefined analysis_job_ids
        cross_analysis_job_id=cross_analysis_job_id,
        similarity_matrix=video_similarities,
        cross_video_segments=chunking_points,
        metadata={
            "analysis_method": "real_cross_video_analysis",
            "settings": settings,
            "optimal_sequence": optimal_sequence,
            "individual_analyses": {
                str(video_ids[0]): video_1_analysis,
                str(video_ids[1]): video_2_analysis
            },
            "analysis_summary": {
                "total_videos": len(video_ids),
                "total_segments": len(chunking_points),
                "average_similarity": sum(video_similarities[str(video_ids[0])].values()) / len(video_similarities[str(video_ids[0])])
            }
        }
    )
    
    logger.info(f"[CROSS ANALYSIS] Completed cross-video analysis for project {project_id}")
    logger.info(f"[CROSS ANALYSIS] Generated {len(chunking_points)} chunking points")
    logger.info(f"[CROSS ANALYSIS] Similarity matrix: {video_similarities}")
    
    return cross_analysis_result

async def _trigger_editing_for_project(project_id: str):
    """
    Automatically trigger editing for a project using create_robust_25_second_video.py.
    This function is called by the worker when analysis jobs complete.
    """
    try:
        from uuid import UUID
        project_uuid = UUID(project_id)
        
        print(f"🔧 [DEBUG] _trigger_editing_for_project called with project_id: {project_id}")
        logger.info(f"[AUTO EDITING] Starting editing for project {project_id}")
        
        # Get the multi-video manager
        print(f"🔧 [DEBUG] Getting multi-video manager...")
        multi_video_manager = await get_multi_video_manager()
        print(f"🔧 [DEBUG] Getting project {project_uuid}...")
        project = await multi_video_manager.get_project(project_uuid)
        print(f"🔧 [DEBUG] Project retrieved: {project}")
        
        if not project:
            print(f"🔧 [DEBUG] Project not found!")
            logger.error(f"[AUTO EDITING] Project {project_id} not found")
            return
        
        print(f"🔧 [DEBUG] Project found: {project.name}, video_ids: {project.video_ids}")
        logger.info(f"[AUTO EDITING] Project found: {project.name}, video_ids: {project.video_ids}")
        
        # Check if editing is already in progress or completed
        print(f"🔧 [DEBUG] Checking if editing job exists: {project.editing_job}")
        if project.editing_job:
            print(f"🔧 [DEBUG] Editing job already exists, returning")
            logger.info(f"[AUTO EDITING] Editing job already exists for project {project_id}: {project.editing_job}")
            return
        
        # Create editing job using service manager
        from app.services import get_service_manager
        service_manager = get_service_manager()
        job_queue = await service_manager.get_redis()
        
        logger.info(f"[AUTO EDITING] Service manager and job queue obtained")
        
        # Create editing job metadata
        editing_metadata = {
            "job_type": "multi_video_editing",
            "project_id": str(project_id),
            "video_ids": [str(vid) for vid in project.video_ids],
            "use_new_workflow": True,  # Flag to use create_robust_25_second_video.py
            "workflow_type": "gemini_direct"  # Use Gemini directly
        }
        
        logger.info(f"[AUTO EDITING] Editing metadata: {editing_metadata}")
        
        # Create the editing job using multi-video method
        logger.info(f"[AUTO EDITING] Creating multi-video editing job...")
        print(f"🔧 [DEBUG] About to call create_multi_video_editing_job with video_id: {project.video_ids[0]}")
        editing_job = await job_queue.create_multi_video_editing_job(
            video_id=project.video_ids[0],  # Use first video as primary
            custom_settings=editing_metadata
        )
        print(f"🔧 [DEBUG] create_multi_video_editing_job returned: {editing_job}")
        editing_job_id = editing_job.job_id
        print(f"🔧 [DEBUG] editing_job_id: {editing_job_id}")
        logger.info(f"[AUTO EDITING] Created editing job: {editing_job_id}")
        
        if editing_job_id:
            # Enqueue the editing job
            logger.info(f"[AUTO EDITING] Enqueuing editing job {editing_job_id}...")
            enqueue_result = job_queue.enqueue_job(editing_job)
            logger.info(f"[AUTO EDITING] Enqueue result: {enqueue_result}")
            
            if enqueue_result:
                # Set the editing job in the project
                logger.info(f"[AUTO EDITING] Setting editing job in project...")
                await multi_video_manager.set_editing_job(project_uuid, editing_job_id)
                logger.info(f"[AUTO EDITING] Editing job {editing_job_id} created and set for project {project_id}")
            else:
                logger.error(f"[AUTO EDITING] Failed to enqueue editing job {editing_job_id}")
        else:
            logger.error(f"[AUTO EDITING] Failed to create editing job for project {project_id}")
            
    except Exception as e:
        logger.error(f"[AUTO EDITING] Error triggering editing for project {project_id}: {e}")
        import traceback
        logger.error(f"[AUTO EDITING] Traceback: {traceback.format_exc()}")

async def _trigger_cross_analysis_for_project(project_id: str):
    """
    Automatically trigger cross-analysis for a project when all individual analysis jobs are completed.
    This function is called by the worker when an analysis job completes.
    """
    try:
        from uuid import UUID
        project_uuid = UUID(project_id)
        
        logger.info(f"[AUTO CROSS-ANALYSIS] Starting cross-analysis for project {project_id}")
        
        # Get the multi-video manager
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_uuid)
        
        if not project:
            logger.error(f"[AUTO CROSS-ANALYSIS] Project {project_id} not found")
            return
        
        # Check if cross-analysis is already in progress or completed
        if project.cross_analysis_job:
            logger.info(f"[AUTO CROSS-ANALYSIS] Cross-analysis job already exists for project {project_id}: {project.cross_analysis_job}")
            
            # Get the job queue
            from app.services import get_service_manager
            job_queue = await get_service_manager().get_redis()
            
            # Check if we need to update the existing cross-analysis job with new analysis jobs
            cross_analysis_job = job_queue._load_job_from_redis(project.cross_analysis_job)
            if cross_analysis_job and cross_analysis_job.metadata:
                current_analysis_job_ids = cross_analysis_job.metadata.get("analysis_job_ids", [])
                project_analysis_job_ids = [str(job_id) for job_id in project.analysis_jobs]
                
                # Check if we have new analysis jobs that aren't in the cross-analysis job
                missing_analysis_jobs = [job_id for job_id in project_analysis_job_ids if job_id not in current_analysis_job_ids]
                
                if missing_analysis_jobs:
                    logger.info(f"[AUTO CROSS-ANALYSIS] Found {len(missing_analysis_jobs)} new analysis jobs, updating cross-analysis job")
                    
                    # Update the cross-analysis job metadata with all analysis job IDs
                    cross_analysis_job.metadata["analysis_job_ids"] = project_analysis_job_ids
                    cross_analysis_job.metadata["video_ids"] = [str(vid) for vid in project.video_ids]
                    
                    # Reset the job status to pending so it can be reprocessed
                    cross_analysis_job.status = ProcessingStatus.PENDING
                    cross_analysis_job.progress = 0
                    cross_analysis_job.error_message = ""
                    
                    # Save the updated job
                    job_queue._save_job_to_redis(cross_analysis_job)
                    
                    # Re-enqueue the job for processing
                    job_queue.queue.enqueue(
                        "app.job_queue.worker.process_cross_analysis_job_standalone",
                        str(project.cross_analysis_job),
                        job_id=str(project.cross_analysis_job)
                    )
                    
                    logger.info(f"[AUTO CROSS-ANALYSIS] Updated and re-enqueued cross-analysis job with all {len(project_analysis_job_ids)} analysis jobs")
                    return
                else:
                    logger.info(f"[AUTO CROSS-ANALYSIS] Cross-analysis job already has all analysis jobs, skipping")
                    return
            else:
                logger.info(f"[AUTO CROSS-ANALYSIS] Cross-analysis job not found or has no metadata, proceeding with new creation")
        
        # Create cross-analysis job directly (since the endpoint is disabled)
        logger.info(f"[AUTO CROSS-ANALYSIS] Creating cross-analysis job directly for project {project_id}")
        
        # Get job queue
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        
        # Get all completed analysis jobs
        analysis_job_ids = []
        analysis_results = []
        for job_id in project.analysis_jobs:
            job = job_queue._load_job_from_redis(job_id)
            if job and job.status == ProcessingStatus.COMPLETED and job.analysis_result:
                analysis_job_ids.append(job.job_id)
                analysis_results.append(job.analysis_result)
        
        if len(analysis_results) < len(project.video_ids):
            logger.warning(f"[AUTO CROSS-ANALYSIS] Not enough analysis results: {len(analysis_results)}/{len(project.video_ids)}")
            return
        
        # Create cross-analysis job
        cross_analysis_job = ProcessingJob(
            job_id=uuid4(),
            video_id=project.video_ids[0],  # Use first video as primary
            status=ProcessingStatus.PENDING,
            template_type=TemplateType.BEAT_MATCH,
            quality_preset=QualityPreset.HIGH,
            metadata={
                "job_type": "cross_video_analysis",
                "project_id": str(project_id),
                "video_ids": [str(vid) for vid in project.video_ids],
                "analysis_job_ids": [str(job_id) for job_id in analysis_job_ids],
                "cross_analysis_settings": {
                    "enableCrossAnalysis": True,
                    "similarityThreshold": 0.7,
                    "chunkingStrategy": "scene"
                }
            }
        )
        
        # Enqueue the cross-analysis job
        enqueue_success = job_queue.enqueue_job(cross_analysis_job)
        if not enqueue_success:
            logger.error("[AUTO CROSS-ANALYSIS] Failed to enqueue cross-analysis job")
            return
        
        # Save the job
        job_queue._save_job_to_redis(cross_analysis_job)
        
        # Set the cross-analysis job in the project
        await multi_video_manager.set_cross_analysis_job(
            project_id=project_uuid,
            cross_analysis_job_id=cross_analysis_job.job_id
        )
        
        logger.info(f"[AUTO CROSS-ANALYSIS] Cross-analysis job {cross_analysis_job.job_id} created and enqueued successfully for project {project_id}")
        return
        
    except Exception as e:
        logger.error(f"[AUTO CROSS-ANALYSIS] Error triggering cross-analysis for project {project_id}: {e}")


async def _trigger_llm_recommendation_after_cross_analysis(project_id: str):
    """
    Automatically trigger LLM recommendation after cross-analysis completes.
    This function is called by the worker when cross-analysis job completes.
    """
    try:
        from uuid import UUID
        project_uuid = UUID(project_id)
        
        logger.info(f"[AUTO LLM TRIGGER] Starting LLM recommendation for project {project_id}")
        
        # Get the multi-video manager
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_uuid)
        
        if not project:
            logger.error(f"[AUTO LLM TRIGGER] Project {project_id} not found")
            return
        
        # Create default edit settings for automatic LLM trigger
        edit_settings = {
            "style_preferences": {
                "energy_level": "high",
                "pacing": "fast"
            },
            "quality_preset": "high",
            "target_duration": None,
            "cross_video_settings": {
                "enableCrossAnalysis": True,
                "similarityThreshold": 0.5,
                "chunkingStrategy": "scene"
            }
        }
        
        # Call the LLM recommendation logic
        logger.info(f"[AUTO LLM TRIGGER] About to call _generate_llm_recommendation_for_project for project {project_id}")
        try:
            await _generate_llm_recommendation_for_project(project_uuid, edit_settings)
            logger.info(f"[AUTO LLM TRIGGER] LLM recommendation completed for project {project_id}")
        except Exception as llm_error:
            logger.error(f"[AUTO LLM TRIGGER] LLM recommendation failed: {llm_error}")
            import traceback
            logger.error(f"[AUTO LLM TRIGGER] LLM recommendation traceback: {traceback.format_exc()}")
            raise llm_error
        
    except Exception as e:
        logger.error(f"[AUTO LLM TRIGGER] Error triggering LLM recommendation: {e}")
        import traceback
        logger.error(f"[AUTO LLM TRIGGER] Full traceback: {traceback.format_exc()}")
        # Don't re-raise - we want this to fail gracefully and not break the worker


async def _generate_llm_recommendation_for_project(project_id: UUID, edit_settings: Dict[str, Any]):
    """
    Generate LLM recommendation for a project with given edit settings.
    """
    try:
        logger.info(f"[LLM RECOMMENDATION] Generating LLM recommendation for project {project_id}")
        
        # Get the multi-video manager
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_id)
        
        if not project:
            logger.error(f"[LLM RECOMMENDATION] Project {project_id} not found")
            return
        
        # Get analysis results using the project's analysis_jobs array
        analysis_results = []
        analysis_job_ids = []
        
        job_queue = await get_job_queue()
        for analysis_job_id in project.analysis_jobs:
            analysis_job = job_queue._load_job_from_redis(analysis_job_id)
            if analysis_job and analysis_job.analysis_result:
                analysis_results.append(analysis_job.analysis_result)
                analysis_job_ids.append(analysis_job.job_id)
                logger.info(f"[LLM RECOMMENDATION] Loaded analysis job {analysis_job_id} for video {analysis_job.video_id}")
            else:
                logger.warning(f"[LLM RECOMMENDATION] Analysis job {analysis_job_id} not found or has no result")
        
        if not analysis_results:
            logger.error(f"[LLM RECOMMENDATION] No analysis results found for project {project_id}")
            return
        
        # Get cross-analysis result
        cross_analysis_job = job_queue._load_job_from_redis(project.cross_analysis_job)
        cross_analysis_result = None
        
        if cross_analysis_job and cross_analysis_job.metadata.get('cross_analysis_result'):
            cross_analysis_result = cross_analysis_job.metadata.get('cross_analysis_result')
        elif cross_analysis_job and cross_analysis_job.analysis_result:
            cross_analysis_result = cross_analysis_job.analysis_result
        elif cross_analysis_job and cross_analysis_job.metadata.get('cross_video_analysis'):
            cross_analysis_result = cross_analysis_job.metadata.get('cross_video_analysis')
        
        if not cross_analysis_result:
            logger.error(f"[LLM RECOMMENDATION] No cross-analysis result found for project {project_id}")
            return
        
        # Generate LLM recommendation
        logger.info(f"[LLM RECOMMENDATION] Generating LLM recommendation with {len(analysis_results)} analysis results")
        
        from app.editor.enhanced_llm_editor import create_enhanced_llm_editor
        from app.models.schemas import EditStyle
        
        # Create enhanced LLM editor
        enhanced_editor = create_enhanced_llm_editor("openai")
        
        # Determine style based on edit settings
        style_str = "tiktok"  # default
        if edit_settings.get("style_preferences", {}).get("energy_level") == "medium":
            style_str = "youtube"
        elif edit_settings.get("style_preferences", {}).get("energy_level") == "low":
            style_str = "cinematic"
        
        style = EditStyle(style_str)
        
        # Calculate target duration based on actual video durations
        total_video_duration = sum(analysis.duration for analysis in analysis_results)
        target_duration = edit_settings.get("target_duration") or total_video_duration
        logger.info(f"[LLM RECOMMENDATION] Calculated target duration: {target_duration}s (from {len(analysis_results)} videos, total duration: {total_video_duration}s)")
        
        # Create combined analysis context
        combined_analysis_context = {
            "primary_analysis": analysis_results[0],
            "video_analyses": analysis_results,  # This is what the LLM editor expects
            "all_analysis_results": analysis_results,
            "cross_video_analysis": cross_analysis_result,
            "video_ids": [str(vid) for vid in project.video_ids],
            "analysis_job_ids": [str(job_id) for job_id in analysis_job_ids],
            "cross_analysis_job_id": str(project.cross_analysis_job),
            "edit_settings": edit_settings
        }
        
        # Log detailed breakdown of what we're sending to LLM
        logger.info("🎯 [LLM INPUT] Detailed breakdown of analysis data being sent to LLM:")
        logger.info("=" * 80)
        
        # Log primary analysis
        logger.info("📊 PRIMARY ANALYSIS (Video 1):")
        if isinstance(analysis_results[0], dict):
            primary_analysis = analysis_results[0]
            logger.info(f"   Keys: {list(primary_analysis.keys())}")
            if 'motion_analysis' in primary_analysis:
                motion = primary_analysis['motion_analysis']
                logger.info(f"   Motion Analysis: {motion.get('motion_score', 'N/A')} intensity")
            if 'audio_analysis' in primary_analysis:
                audio = primary_analysis['audio_analysis']
                logger.info(f"   Audio Analysis: {len(audio.get('beats', []))} beats, {audio.get('tempo', 'N/A')} BPM")
            if 'metadata' in primary_analysis:
                meta = primary_analysis['metadata']
                logger.info(f"   Metadata: {meta.get('duration', 'N/A')}s, {meta.get('fps', 'N/A')} FPS")
        else:
            logger.info(f"   Type: {type(analysis_results[0])}")
        
        # Log all analysis results
        logger.info(f"📊 ALL ANALYSIS RESULTS ({len(analysis_results)} videos):")
        for i, analysis in enumerate(analysis_results):
            logger.info(f"   Video {i+1}:")
            if isinstance(analysis, dict):
                logger.info(f"     Keys: {list(analysis.keys())}")
                if 'motion_analysis' in analysis:
                    motion = analysis['motion_analysis']
                    logger.info(f"     Motion: {motion.get('motion_score', 'N/A')} intensity")
                if 'audio_analysis' in analysis:
                    audio = analysis['audio_analysis']
                    logger.info(f"     Audio: {len(audio.get('beats', []))} beats, {audio.get('tempo', 'N/A')} BPM")
                if 'metadata' in analysis:
                    meta = analysis['metadata']
                    logger.info(f"     Metadata: {meta.get('duration', 'N/A')}s, {meta.get('fps', 'N/A')} FPS")
            else:
                logger.info(f"     Type: {type(analysis)}")
        
        # Log cross-analysis
        logger.info("🔗 CROSS-VIDEO ANALYSIS:")
        if hasattr(cross_analysis_result, 'metadata') and isinstance(cross_analysis_result.metadata, dict):
            cross_meta = cross_analysis_result.metadata
            logger.info(f"   Analysis Method: {cross_meta.get('analysis_method', 'N/A')}")
            if 'analysis_summary' in cross_meta:
                summary = cross_meta['analysis_summary']
                logger.info(f"   Summary: {summary.get('total_videos', 'N/A')} videos, {summary.get('total_segments', 'N/A')} segments")
                logger.info(f"   Average Similarity: {summary.get('average_similarity', 'N/A')}")
            if 'individual_analyses' in cross_meta:
                logger.info(f"   Individual Analyses: {len(cross_meta['individual_analyses'])} videos included")
        else:
            logger.info(f"   Type: {type(cross_analysis_result)}")
        
        # Log similarity matrix
        if hasattr(cross_analysis_result, 'similarity_matrix'):
            logger.info("📈 SIMILARITY MATRIX:")
            for vid1, similarities in cross_analysis_result.similarity_matrix.items():
                for vid2, similarity in similarities.items():
                    logger.info(f"   {vid1[:8]} -> {vid2[:8]}: {similarity:.3f}")
        
        # Log chunking points
        if hasattr(cross_analysis_result, 'cross_video_segments'):
            logger.info(f"🎬 CHUNKING POINTS: {len(cross_analysis_result.cross_video_segments)} total")
            for i, point in enumerate(cross_analysis_result.cross_video_segments[:5]):  # Show first 5
                logger.info(f"   Point {i+1}: {point.get('chunk_type', 'N/A')} at {point.get('start_time', 0):.1f}s")
            if len(cross_analysis_result.cross_video_segments) > 5:
                logger.info(f"   ... and {len(cross_analysis_result.cross_video_segments) - 5} more")
        
        logger.info("=" * 80)
        
        logger.info("Generating enhanced editing plan with cross-video context...")
        
        # Generate LLM recommendation
        editing_plan = await enhanced_editor.generate_editing_plan(
            analysis_result=analysis_results[0],
            style=style,
            target_duration=target_duration,  # Use calculated target duration
            multi_video_context=combined_analysis_context
        )
        
        # Log the LLM response in detail
        logger.info("🎯 [LLM RESPONSE] Detailed breakdown of LLM recommendation:")
        logger.info("=" * 80)
        
        # Convert the enhanced plan to a serializable format
        import json
        plan_dict = {
            "style": editing_plan.style,
            "target_duration": editing_plan.target_duration,
            "segments": [
                {
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "effects": segment.effects,
                    "source_video_id": str(segment.source_video_id),  # Convert UUID to string
                    "transition_in": segment.transition_in,
                    "transition_out": segment.transition_out
                }
                for segment in editing_plan.segments
            ]
        }
        
        # Log the complete LLM response
        logger.info(json.dumps(plan_dict, indent=2, default=str))
        
        # Log summary
        logger.info(f"📊 SUMMARY:")
        logger.info(f"   Total segments: {len(editing_plan.segments)}")
        logger.info(f"   Style: {editing_plan.style}")
        logger.info(f"   Target duration: {editing_plan.target_duration}")
        
        # Log segment breakdown
        logger.info(f"🎬 SEGMENT BREAKDOWN:")
        for i, segment in enumerate(editing_plan.segments):
            duration = segment.end_time - segment.start_time
            logger.info(f"   Segment {i+1}: {segment.start_time:.1f}s - {segment.end_time:.1f}s ({duration:.1f}s)")
            logger.info(f"     Effects: {segment.effects}")
            logger.info(f"     Source Video: {segment.source_video_id}")
            logger.info(f"     Transitions: {segment.transition_in} -> {segment.transition_out}")
        
        logger.info("=" * 80)
        
        logger.info(f"[LLM RECOMMENDATION] LLM recommendation generated successfully")
        logger.info(f"[LLM RECOMMENDATION] Plan contains {len(editing_plan.segments)} segments")
        
        # Create editing job with LLM recommendation (using multi-video method)
        logger.info(f"[LLM RECOMMENDATION] About to create multi-video editing job...")
        logger.info(f"[LLM RECOMMENDATION] Project video IDs: {project.video_ids}")
        logger.info(f"[LLM RECOMMENDATION] Analysis job IDs: {analysis_job_ids}")
        logger.info(f"[LLM RECOMMENDATION] Cross analysis job: {project.cross_analysis_job}")
        
        try:
            job_queue = await get_job_queue()
            logger.info(f"[LLM RECOMMENDATION] Got job queue: {job_queue}")
            
            editing_job = await job_queue.create_multi_video_editing_job(
                video_id=project.video_ids[0],  # Use first video as primary
                template_type=TemplateType.BEAT_MATCH,
                custom_settings={
                    "multi_video_project_id": str(project_id),
                    "multi_video_video_ids": [str(vid) for vid in project.video_ids],
                    "multi_video_analysis_jobs": [str(job_id) for job_id in analysis_job_ids],
                    "multi_video_cross_analysis_job": str(project.cross_analysis_job),
                    "edit_settings": edit_settings
                },
                quality_preset=QualityPreset.HIGH,
                analysis_result=analysis_results[0]  # Use first analysis result
            )
            logger.info(f"[LLM RECOMMENDATION] Successfully created editing job: {editing_job.job_id}")
            
        except Exception as e:
            logger.error(f"[LLM RECOMMENDATION] Failed to create editing job: {e}")
            import traceback
            logger.error(f"[LLM RECOMMENDATION] Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to create editing job: {e}")
        
        # Store the LLM plan in the same location as single video workflow
        editing_job.metadata["enhanced_llm_plan_json"] = json.dumps(plan_dict)
        job_queue._save_job_to_redis(editing_job)
        
        logger.info(f"[LLM RECOMMENDATION] Editing job created: {editing_job.job_id}")
        
        # Enqueue the editing job for processing
        logger.info(f"[LLM RECOMMENDATION] About to enqueue editing job {editing_job.job_id}...")
        try:
            enqueue_success = job_queue.enqueue_job(editing_job)
            if not enqueue_success:
                logger.error(f"[LLM RECOMMENDATION] Failed to enqueue editing job {editing_job.job_id}")
                raise Exception("Failed to enqueue editing job")
            
            logger.info(f"[LLM RECOMMENDATION] Editing job enqueued successfully: {editing_job.job_id}")
            
        except Exception as e:
            logger.error(f"[LLM RECOMMENDATION] Exception during enqueue: {e}")
            import traceback
            logger.error(f"[LLM RECOMMENDATION] Enqueue traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to enqueue editing job: {e}")
        
        # Update project with editing job
        logger.info(f"[LLM RECOMMENDATION] About to update project with editing job {editing_job.job_id}...")
        try:
            project.editing_job = editing_job.job_id
            await multi_video_manager.update_project(project)
            logger.info(f"[LLM RECOMMENDATION] Project updated successfully with editing job")
            
        except Exception as e:
            logger.error(f"[LLM RECOMMENDATION] Failed to update project: {e}")
            import traceback
            logger.error(f"[LLM RECOMMENDATION] Project update traceback: {traceback.format_exc()}")
            # Don't fail the whole process if project update fails
            logger.warning(f"[LLM RECOMMENDATION] Continuing despite project update failure")
        
        logger.info(f"[LLM RECOMMENDATION] Editing job created and enqueued: {editing_job.job_id}")
        logger.info(f"[LLM RECOMMENDATION] Auto-triggering complete - editing job {editing_job.job_id} is now processing")
        
    except Exception as e:
        logger.error(f"[LLM RECOMMENDATION] Error generating LLM recommendation: {e}")


async def _create_multi_video_editing_job(
    project_id: UUID,
    video_ids: List[UUID],
    analysis_jobs: List[UUID],
    cross_analysis_job: UUID,
    edit_settings: Dict[str, Any]
):
    """Create a multi-video editing job."""
    try:
        from app.services import get_service_manager
        from app.models.schemas import QualityPreset, TemplateType
        from uuid import uuid4
        
        logger.info(f"Creating multi-video editing job for project {project_id}")
        logger.info(f"Video IDs: {video_ids}")
        logger.info(f"Analysis jobs: {analysis_jobs}")
        logger.info(f"Cross analysis job: {cross_analysis_job}")
        
        # Test service manager first
        try:
            logger.info("Getting service manager...")
            service_manager = get_service_manager()
            logger.info(f"Got service manager: {service_manager}")
            logger.info("Getting Redis from service manager...")
            job_queue = await service_manager.get_redis()
            logger.info(f"Got job queue: {job_queue}")
            logger.info(f"Job queue type: {type(job_queue)}")
            logger.info(f"Job queue methods: {dir(job_queue)}")
        except Exception as e:
            logger.error(f"Service manager error: {e}")
            import traceback
            logger.error(f"Service manager traceback: {traceback.format_exc()}")
            raise Exception(f"Service manager error: {e}")
        
        # Get the analysis results from the individual analysis jobs
        analysis_results = []
        for analysis_job_id in analysis_jobs:
            analysis_job = job_queue._load_job_from_redis(analysis_job_id)
            if analysis_job and analysis_job.analysis_result:
                analysis_results.append(analysis_job.analysis_result)
                logger.info(f"Found analysis result for job {analysis_job_id}")
            else:
                logger.warning(f"No analysis result found for job {analysis_job_id}")
        
        if not analysis_results:
            raise Exception("No analysis results found for any of the analysis jobs")
        
        # Use the first analysis result as the primary one for the editing job
        primary_analysis_result = analysis_results[0]
        logger.info(f"Using analysis result from job {analysis_jobs[0]} as primary")
        
        # Create the editing job with the analysis result
        editing_job = ProcessingJob(
            job_id=uuid4(),
            video_id=video_ids[0],  # Use first video as primary
            status=ProcessingStatus.PENDING,
            template_type=TemplateType.BEAT_MATCH,
            quality_preset=QualityPreset.HIGH,
            analysis_result=primary_analysis_result,  # Attach the analysis result
            metadata={
                "job_type": "multi_video_editing",
                "project_id": str(project_id),
                "video_ids": [str(vid) for vid in video_ids],
                "analysis_job_ids": [str(job_id) for job_id in analysis_jobs],
                "cross_analysis_job_id": str(cross_analysis_job),
                "edit_settings": edit_settings,
                "all_analysis_results": [str(result) for result in analysis_results]  # Store all results
            }
        )
        
        logger.info(f"Created editing job object: {editing_job.job_id}")
        logger.info(f"Attached analysis result: {editing_job.analysis_result is not None}")
        
        # Save and enqueue the job
        try:
            # Save job to Redis first
            save_result = job_queue._save_job_to_redis(editing_job)
            logger.info(f"Save result: {save_result}")
            
            if not save_result:
                raise Exception("Failed to save multi-video editing job to Redis")
            
            # Enqueue job in RQ queue
            enqueue_result = job_queue.enqueue_job(editing_job)
            logger.info(f"Enqueue result: {enqueue_result}")
            
            if not enqueue_result:
                raise Exception("Failed to enqueue multi-video editing job in RQ queue")
            
            logger.info(f"Created multi-video editing job {editing_job.job_id} for project {project_id}")
            return editing_job
            
        except Exception as e:
            logger.error(f"Job creation/enqueue failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise Exception(f"Job creation/enqueue failed: {e if e else 'Unknown error'}")
    except Exception as e:
        logger.error(f"Error in _create_multi_video_editing_job: {e}")
        raise e 

@video_router.get("/queue/status")
async def get_queue_status():
    """Get comprehensive queue status for monitoring."""
    try:
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        status = job_queue.get_queue_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

@video_router.post("/queue/recover")
async def recover_stuck_jobs():
    """Manually trigger recovery of stuck jobs."""
    try:
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        
        # Check for stuck jobs
        stuck_jobs = job_queue.check_stuck_jobs()
        if not stuck_jobs:
            return {"message": "No stuck jobs found", "recovered": 0}
        
        # Attempt recovery
        recovered_count = job_queue.recover_stuck_jobs()
        
        return {
            "message": f"Recovery attempt completed",
            "stuck_jobs_found": len(stuck_jobs),
            "recovered": recovered_count,
            "stuck_job_ids": [str(job.job_id) for job in stuck_jobs]
        }
    except Exception as e:
        logger.error(f"Failed to recover stuck jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to recover stuck jobs: {str(e)}")

@video_router.post("/queue/cleanup")
async def cleanup_failed_jobs():
    """Manually trigger cleanup of failed jobs."""
    try:
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        
        cleaned_count = job_queue.cleanup_failed_jobs()
        
        return {
            "message": f"Cleanup completed",
            "cleaned": cleaned_count
        }
    except Exception as e:
        logger.error(f"Failed to cleanup failed jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup failed jobs: {str(e)}")

@video_router.post("/queue/cleanup-all")
async def cleanup_all_failed_jobs():
    """Aggressively clean up ALL failed jobs (use with caution)."""
    try:
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        
        cleaned_count = job_queue.cleanup_all_failed_jobs()
        
        return {
            "message": f"Aggressive cleanup completed",
            "cleaned": cleaned_count,
            "warning": "This removed ALL failed jobs regardless of age or retry count"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup all failed jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup all failed jobs: {str(e)}")

@video_router.get("/queue/health")
async def get_queue_health():
    """Get queue health status."""
    try:
        from app.services import get_service_manager
        job_queue = await get_service_manager().get_redis()
        
        # Perform health check
        is_healthy = await job_queue.health_check()
        
        # Get detailed status
        status = job_queue.get_queue_status()
        
        return {
            "healthy": is_healthy,
            "status": status
        }
    except Exception as e:
        logger.error(f"Failed to get queue health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue health: {str(e)}")

@multi_video_router.post("/projects/{project_id}/test-llm-timeline")
async def test_llm_timeline_generation(
    project_id: UUID,
    request: MultiVideoEditRequest
) -> BaseResponse:
    """
    Test LLM timeline generation for multi-video project using existing structure.
    
    Args:
        project_id: Multi-video project ID
        request: Multi-video edit request
        
    Returns:
        BaseResponse: Test result
    """
    try:
        logger.info(f"🧪 [API] Testing LLM timeline generation for project {project_id}")
        
        # Get project
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        logger.info(f"🧪 [API] Found project: {project.name} with {len(project.video_ids)} videos")
        
        # Get enhanced LLM editor
        enhanced_llm_editor = get_enhanced_llm_editor()
        
        # Create mock analysis result for testing
        mock_analysis_result = VideoAnalysisResult(
            video_id=project.video_ids[0],  # Use first video for testing
            duration=30.0,  # Mock duration
            fps=30.0,
            resolution=(1920, 1080),
            beat_detection=BeatDetectionResult(
                timestamps=[0.0, 1.0, 2.0],
                confidence_scores=[0.8, 0.9, 0.7],
                bpm=120.0,
                energy_levels=[0.6, 0.8, 0.5]
            ),
            motion_analysis=MotionAnalysisResult(
                motion_spikes=[0.5, 1.5, 2.5],
                motion_intensities=[0.7, 0.9, 0.6],
                scene_changes=[1.0, 2.0],
                scene_confidence=[0.8, 0.9],
                motion_score=0.75,
                activity_level="medium"
            ),
            audio_analysis=AudioAnalysisResult(
                volume_levels=[0.6, 0.8, 0.7],
                silence_periods=[[0.0, 0.1]],
                audio_peaks=[0.5, 1.5],
                frequency_analysis={"low": [0.3], "mid": [0.5], "high": [0.2]}
            )
        )
        
        # Create multi-video context
        multi_video_context = {
            "project_id": str(project_id),
            "video_ids": [str(vid_id) for vid_id in project.video_ids],
            "total_videos": len(project.video_ids),
            "cross_analysis_completed": True,
            "similarity_matrix": {},  # Mock similarity data
            "cross_video_segments": []  # Mock cross-video segments
        }
        
        # Generate LLM timeline assignment
        logger.info("🧪 [API] Generating LLM timeline assignment...")
        timeline_assignment = await enhanced_llm_editor._generate_multi_video_timeline_assignment(
            analysis_result=mock_analysis_result,
            style=EditStyle.CINEMATIC,
            target_duration=request.target_duration,
            multi_video_context=multi_video_context
        )
        
        # Calculate actual estimated duration from segments
        segments = timeline_assignment.get("segments", [])
        actual_duration = sum([seg.get("end_time", 0) - seg.get("start_time", 0) for seg in segments])
        
        # Create response with test results
        test_result = {
            "success": True,
            "message": "LLM timeline test completed successfully",
            "details": {
                "segments_count": len(segments),
                "confidence_score": timeline_assignment.get("confidence_score", 0.8),
                "estimated_duration": actual_duration,
                "llm_estimated_duration": timeline_assignment.get("estimated_duration", actual_duration),
                "overall_strategy": timeline_assignment.get("overall_strategy", "Cinematic multi-video compilation with smooth transitions"),
                "duration_guidance": "Content-driven duration based on best moments",
                "segment_assignments": [
                    {
                        "source_video_id": seg.get("source_video_id", seg.get("video_id", "unknown")),
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "duration": seg.get("end_time", 0) - seg.get("start_time", 0),
                        "llm_reasoning": seg.get("llm_reasoning", "Selected for visual impact"),
                        "segment_order": seg.get("segment_order", i)
                    }
                    for i, seg in enumerate(segments)
                ]
            }
        }
        
        logger.info(f"🧪 [API] LLM timeline test completed with {test_result['details']['segments_count']} segments")
        return BaseResponse(**test_result)
        
    except Exception as e:
        logger.error(f"❌ [API] Failed to test LLM timeline generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test timeline generation: {str(e)}")


@multi_video_router.get("/projects/{project_id}/llm-test-result")
async def get_llm_test_result(project_id: UUID) -> BaseResponse:
    """
    Get LLM timeline test result for project.
    
    Args:
        project_id: Multi-video project ID
        
    Returns:
        BaseResponse: Test result
    """
    try:
        logger.info(f"📋 [API] Getting LLM test result for project {project_id}")
        
        # Get project
        multi_video_manager = get_multi_video_manager()
        project = await multi_video_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get test result from metadata
        test_result = project.metadata.get('llm_test_result')
        if not test_result:
            raise HTTPException(status_code=404, detail="No LLM test result found")
        
        return BaseResponse(
            message="LLM test result retrieved successfully",
            details=test_result
        )
        
    except Exception as e:
        logger.error(f"❌ [API] Failed to get LLM test result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test result: {str(e)}")

@video_router.get("/{video_id}/duration")
async def get_video_duration(
    video_id: UUID,
    storage_client=Depends(get_storage_client)
):
    """
    Get the actual duration of a video file for debugging.
    """
    try:
        import tempfile
        import os
        from moviepy import VideoFileClip
        
        # Get the original video file from S3
        original_key = f"uploads/{video_id}.mp4"
        temp_input_path = os.path.join(tempfile.gettempdir(), f"duration_check_{video_id}.mp4")
        
        try:
            # Download original video to temp file
            storage_client.s3_client.download_file(
                storage_client.bucket_name,
                original_key,
                temp_input_path
            )
            logger.info(f"📥 Downloaded video for duration check: {temp_input_path}")
        except Exception as e:
            logger.error(f"❌ Failed to download video for duration check: {e}")
            raise HTTPException(status_code=404, detail="Video not found")
        
        try:
            video_clip = VideoFileClip(temp_input_path)
            video_duration = video_clip.duration
            video_fps = video_clip.fps
            video_size = (video_clip.w, video_clip.h)
            file_size = os.path.getsize(temp_input_path)
            video_clip.close()
            
            # Clean up temp file
            try:
                os.remove(temp_input_path)
            except:
                pass
            
            return {
                "video_id": str(video_id),
                "duration": video_duration,
                "fps": video_fps,
                "resolution": {"width": video_size[0], "height": video_size[1]},
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze video duration: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to analyze video: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Duration check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Duration check failed: {str(e)}")

@video_router.post("/{video_id}/download-with-custom-effects")
async def download_video_with_custom_effects(
    video_id: UUID,
    custom_effects_request: dict,
    storage_client=Depends(get_storage_client)
):
    """
    Download video with custom effects using the optimized SimpleVideoRenderer.
    
    This endpoint now renders the video with the enhanced shader library
    that provides fast, high-quality effects.
    """
    try:
        logger.info(f"🎬 Starting custom effects rendering for video {video_id}")
        logger.info(f"📊 Custom effects request: {custom_effects_request}")
        
        # Import the optimized renderer
        from app.editor.renderer_simple import SimpleVideoRenderer
        from app.models.schemas import VideoTimeline, TimelineSegment, EditingTemplate, TemplateType, QualityPreset
        import uuid
        import tempfile
        import os
        
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="custom_effects_")
        
        try:
            # VIDEO SOURCE STRATEGY: Use PROCESSED video only (same as multi-video)
            # No fallback - if it breaks, we'll fix the root cause
            source_key = f"processed/{video_id}_processed.mp4"
            source_path = os.path.join(temp_dir, "source.mp4")
            
            # Download source video
            storage_client.s3_client.download_file(
                storage_client.bucket_name,
                source_key,
                source_path
            )
            
            # Get actual video duration to validate segments (with fallback)
            actual_video_duration = None
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(source_path) as video_clip:
                    actual_video_duration = video_clip.duration
                    logger.info(f"📏 [DURATION] Actual processed video duration: {actual_video_duration:.2f}s")
            except ImportError:
                logger.warning("⚠️ [DURATION] MoviePy not available, using fallback duration validation")
                # Fallback: assume video duration is reasonable (will be validated by renderer)
                actual_video_duration = 999.0  # Large enough to not trigger adjustments
            except Exception as e:
                logger.warning(f"⚠️ [DURATION] Failed to get video duration: {e}, using fallback")
                actual_video_duration = 999.0  # Large enough to not trigger adjustments
            
            # Extract segments and effects from the request
            segments_data = custom_effects_request.get('segments', [])
            quality_preset = custom_effects_request.get('quality_preset', 'high')
            
            # Convert quality preset string to enum
            quality_enum = QualityPreset.HIGH if quality_preset == 'high' else QualityPreset.MEDIUM
            
            # Create VideoTimeline from segments with duration validation
            timeline_segments = []
            adjusted_segments = 0
            
            for i, seg_data in enumerate(segments_data):
                # Convert frontend segment format to our TimelineSegment format
                # Frontend sends tags as an array of effect names
                effects = seg_data.get('tags', [])  # Frontend uses 'tags' for effects
                if not effects:
                    effects = seg_data.get('effects', [])  # Fallback to 'effects'
                
                # Ensure effects is always a list
                if isinstance(effects, str):
                    effects = [effects]
                elif not isinstance(effects, list):
                    effects = []
                
                # Validate and adjust segment timing to fit actual video duration
                original_start = seg_data.get('start', 0)
                original_end = seg_data.get('end', 0)
                
                # Adjust segment timing to fit within actual video duration
                adjusted_start = max(0.0, original_start)
                adjusted_end = min(actual_video_duration, original_end)
                
                # Check if segment needs adjustment
                if adjusted_end != original_end or adjusted_start != original_start:
                    adjusted_segments += 1
                    logger.warning(f"⚠️ [DURATION] Segment {i+1} adjusted: {original_start:.2f}s-{original_end:.2f}s → {adjusted_start:.2f}s-{adjusted_end:.2f}s")
                
                # Skip segments that are completely outside the video duration
                if adjusted_start >= adjusted_end:
                    logger.warning(f"⚠️ [DURATION] Segment {i+1} completely outside video duration ({original_start:.2f}s-{original_end:.2f}s), skipping")
                    continue
                
                logger.info(f"🎯 Segment {i+1}: {adjusted_start:.2f}s - {adjusted_end:.2f}s, Effects: {effects}")
                
                segment = TimelineSegment(
                    segment_id=uuid.uuid4(),
                    start_time=adjusted_start,
                    end_time=adjusted_end,
                    source_video_id=video_id,
                    effects=effects,  # Use ALL effects from frontend
                    transition_in=seg_data.get('transition_in', 'fade_in'),
                    transition_out=seg_data.get('transition_out', 'fade_out'),
                    effectCustomizations=seg_data.get('effectCustomizations', {})
                )
                timeline_segments.append(segment)
            
            if adjusted_segments > 0:
                logger.warning(f"⚠️ [DURATION] {adjusted_segments} segments were adjusted to fit video duration")
            
            if len(timeline_segments) == 0:
                raise HTTPException(status_code=400, detail="No valid segments found after duration validation")
            
            # Create editing template (same as our test script)
            template = EditingTemplate(
                template_id=uuid.uuid4(),
                name="Custom Effects Template",
                template_type=TemplateType.FAST_PACED,
                description="Custom effects applied by user",
                transition_duration=0.3,
                cut_sensitivity=0.8,
                beat_sync_threshold=0.1,
                effects=["motion_blur", "color_grading", "high_contrast"]
            )
            
            # Create VideoTimeline
            timeline = VideoTimeline(
                timeline_id=uuid.uuid4(),
                video_id=video_id,
                segments=timeline_segments,
                template=template,
                total_duration=sum(seg.end_time - seg.start_time for seg in timeline_segments),
                quality_preset=quality_enum
            )
            
            # Initialize the optimized renderer (same as our test script)
            renderer = SimpleVideoRenderer()
            
            # Set output path
            output_filename = f"custom_effects_{video_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Render the video with optimized shaders (this is what we tested!)
            logger.info(f"🚀 Rendering video with optimized SimpleVideoRenderer...")
            logger.info(f"📊 Timeline segments: {len(timeline_segments)}")
            logger.info(f"🎨 Effects to apply: {[effect for seg in timeline_segments for effect in seg.effects]}")
            
            success = await renderer.render_video(
                video_path=source_path,
                timeline=timeline,
                output_path=output_path,
                quality_preset=quality_enum
            )
            
            if not success:
                raise Exception("Video rendering failed - check renderer logs for details")
            
            # Upload the rendered video to S3
            storage_key = f"custom_renders/{output_filename}"
            storage_url = await storage_client.upload_file(
                file_path=output_path,
                file_key=storage_key,
                content_type="video/mp4",
                metadata={
                    'video_id': str(video_id),
                    'render_timestamp': datetime.utcnow().isoformat(),
                    'effects_applied': [effect for seg in timeline_segments for effect in seg.effects],
                    'quality_preset': quality_preset
                }
            )
            
            # Create download URL
            download_url = await storage_client.create_download_url(storage_key)
            
            logger.info(f"✅ Custom effects video rendered and uploaded successfully")
            logger.info(f"📁 Output: {storage_key}")
            logger.info(f"🔗 Download URL: {download_url}")
            
            return JSONResponse({
                "success": True,
                "download_url": str(download_url),
                "filename": output_filename,
                "video_id": str(video_id),
                "message": "Video rendered with optimized shader effects",
                "effects_applied": [effect for seg in timeline_segments for effect in seg.effects]
            })
            
        finally:
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 Temporary directory cleaned up: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up temp directory: {cleanup_error}")
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ Custom effects rendering failed for video {video_id}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception message: {str(e)}")
        logger.error(f"❌ Full traceback:")
        logger.error(tb)
        raise HTTPException(status_code=500, detail=f"Custom effects rendering failed: {str(e)}")

@video_router.get("/{video_id}/download-rendered/{render_job_id}")
async def download_rendered_video(
    video_id: UUID,
    render_job_id: str,
    storage_client=Depends(get_storage_client)
):
    """
    Download a rendered video file by render job ID.
    This endpoint streams the file directly to avoid all CORS and SSL issues.
    """
    try:
        # Construct the S3 key for the rendered video
        output_key = f"custom_renders/custom_effects_{video_id}_{render_job_id}.mp4"
        
        # Check if the file exists in S3
        try:
            storage_client.s3_client.head_object(Bucket=storage_client.bucket_name, Key=output_key)
        except Exception as e:
            logger.error(f"❌ Rendered video not found: {output_key}")
            raise HTTPException(status_code=404, detail="Rendered video not found")
        
        # Download the file from S3 to a temporary location
        import tempfile
        import os
        temp_file_path = os.path.join(tempfile.gettempdir(), f"download_{render_job_id}.mp4")
        
        try:
            storage_client.s3_client.download_file(
                storage_client.bucket_name,
                output_key,
                temp_file_path
            )
            
            # Stream the file directly to the client
            from fastapi.responses import FileResponse
            return FileResponse(
                path=temp_file_path,
                media_type="video/mp4",
                filename=f"custom_effects_{video_id}.mp4"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to download file from S3: {e}")
            raise HTTPException(status_code=500, detail="Failed to download file from S3")
        
    except Exception as e:
        logger.error(f"❌ Failed to serve video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve video file: {str(e)}")

# Add this endpoint for multi-video projects
@multi_video_router.post("/projects/{project_id}/download-with-custom-effects")
async def download_multi_video_with_custom_effects(
    project_id: UUID,
    custom_effects_request: dict,
    storage_client=Depends(get_storage_client)
):
    """
    Download multi-video project with custom effects using the optimized SimpleVideoRenderer.
    
    This endpoint now renders the multi-video project with the enhanced shader library
    that provides fast, high-quality effects.
    
    VIDEO SOURCE STRATEGY: Uses PROCESSED video first, then fallback to original (after preview page)
    """
    try:
        logger.info(f"🎬 Starting custom effects rendering for multi-video project {project_id}")
        logger.info(f"📊 Custom effects request: {custom_effects_request}")
        
        from app.services import get_service_manager
        from app.editor.renderer_simple import SimpleVideoRenderer
        from app.models.schemas import VideoTimeline, TimelineSegment, EditingTemplate, TemplateType, QualityPreset
        import uuid
        import tempfile
        import os
        
        # Get project data to find video IDs
        from app.services.multi_video_manager import get_multi_video_manager
        project_manager = await get_multi_video_manager()
        project = await project_manager.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if not project.video_ids or len(project.video_ids) == 0:
            raise HTTPException(status_code=404, detail="No videos found in project")
        
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="multi_video_custom_effects_")
        
        try:
            # VIDEO SOURCE STRATEGY: Use PROCESSED video only (after preview page)
            # No fallback - if it breaks, we'll fix the root cause
            
            # Get the specific video ID to use from the request
            specific_video_id = custom_effects_request.get('video_id')
            if not specific_video_id:
                # Use first video if no specific video ID provided
                specific_video_id = project.video_ids[0]
                logger.info(f"🎯 No specific video_id provided, using first video: {specific_video_id}")
            else:
                logger.info(f"🎯 Using specific video ID: {specific_video_id}")
            
            # Download PROCESSED video only - no fallback
            processed_key = f"processed/{specific_video_id}_processed.mp4"
            temp_input_path = os.path.join(temp_dir, f"processed_{specific_video_id}.mp4")
            
            logger.info(f"📥 [AFTER PREVIEW] Downloading PROCESSED video: {processed_key}")
            
            storage_client.s3_client.download_file(
                storage_client.bucket_name,
                processed_key,
                temp_input_path
            )
            logger.info(f"✅ [AFTER PREVIEW] Downloaded PROCESSED video: {processed_key}")
            
            # Get actual video duration to validate segments (with fallback)
            actual_video_duration = None
            try:
                from moviepy.editor import VideoFileClip
                with VideoFileClip(temp_input_path) as video_clip:
                    actual_video_duration = video_clip.duration
                    logger.info(f"📏 [DURATION] Actual processed video duration: {actual_video_duration:.2f}s")
            except ImportError:
                logger.warning("⚠️ [DURATION] MoviePy not available, using fallback duration validation")
                # Fallback: assume video duration is reasonable (will be validated by renderer)
                actual_video_duration = 999.0  # Large enough to not trigger adjustments
            except Exception as e:
                logger.warning(f"⚠️ [DURATION] Failed to get video duration: {e}, using fallback")
                actual_video_duration = 999.0  # Large enough to not trigger adjustments
            
            # Extract segments and effects from the request
            segments_data = custom_effects_request.get('segments', [])
            quality_preset = custom_effects_request.get('quality_preset', 'high')
            
            # Convert quality preset string to enum
            quality_enum = QualityPreset.HIGH if quality_preset == 'high' else QualityPreset.MEDIUM
            
            # Create VideoTimeline from segments with duration validation
            timeline_segments = []
            adjusted_segments = 0
            
            for i, seg_data in enumerate(segments_data):
                # Convert frontend segment format to our TimelineSegment format
                # Frontend sends tags as an array of effect names
                effects = seg_data.get('tags', [])  # Frontend uses 'tags' for effects
                if not effects:
                    effects = seg_data.get('effects', [])  # Fallback to 'effects'
                
                # Ensure effects is always a list
                if isinstance(effects, str):
                    effects = [effects]
                elif not isinstance(effects, list):
                    effects = []
                
                # Validate and adjust segment timing to fit actual video duration
                original_start = seg_data.get('start', 0)
                original_end = seg_data.get('end', 0)
                
                # Adjust segment timing to fit within actual video duration
                adjusted_start = max(0.0, original_start)
                adjusted_end = min(actual_video_duration, original_end)
                
                # Check if segment needs adjustment
                if adjusted_end != original_end or adjusted_start != original_start:
                    adjusted_segments += 1
                    logger.warning(f"⚠️ [DURATION] Segment {i+1} adjusted: {original_start:.2f}s-{original_end:.2f}s → {adjusted_start:.2f}s-{adjusted_end:.2f}s")
                
                # Skip segments that are completely outside the video duration
                if adjusted_start >= adjusted_end:
                    logger.warning(f"⚠️ [DURATION] Segment {i+1} completely outside video duration ({original_start:.2f}s-{original_end:.2f}s), skipping")
                    continue
                
                logger.info(f"🎯 Multi-video Segment {i+1}: {adjusted_start:.2f}s - {adjusted_end:.2f}s, Effects: {effects}")
                
                segment = TimelineSegment(
                    segment_id=uuid.uuid4(),
                    start_time=adjusted_start,
                    end_time=adjusted_end,
                    source_video_id=specific_video_id,
                    effects=effects,  # Use ALL effects from frontend
                    transition_in=seg_data.get('transition_in', 'fade_in'),
                    transition_out=seg_data.get('transition_out', 'fade_out'),
                    effectCustomizations=seg_data.get('effectCustomizations', {})
                )
                timeline_segments.append(segment)
            
            if adjusted_segments > 0:
                logger.warning(f"⚠️ [DURATION] {adjusted_segments} segments were adjusted to fit video duration")
            
            if len(timeline_segments) == 0:
                raise HTTPException(status_code=400, detail="No valid segments found after duration validation")
            
            # Create editing template (same as our test script)
            template = EditingTemplate(
                template_id=uuid.uuid4(),
                name="Multi-Video Custom Effects Template",
                template_type=TemplateType.FAST_PACED,
                description="Custom effects applied by user to multi-video project",
                transition_duration=0.3,
                cut_sensitivity=0.8,
                beat_sync_threshold=0.1,
                effects=["motion_blur", "color_grading", "high_contrast"]
            )
            
            # Create VideoTimeline
            timeline = VideoTimeline(
                timeline_id=uuid.uuid4(),
                video_id=specific_video_id,  # Use the specific video ID, not first_video_id
                segments=timeline_segments,
                template=template,
                total_duration=sum(seg.end_time - seg.start_time for seg in timeline_segments),
                quality_preset=quality_enum
            )
            
            # Initialize the optimized renderer (same as our test script)
            renderer = SimpleVideoRenderer()
            
            # Set output path
            output_filename = f"multi_video_custom_effects_{project_id}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)
            
            # Render the video with optimized shaders (this is what we tested!)
            logger.info(f"🚀 Rendering multi-video project with optimized SimpleVideoRenderer...")
            logger.info(f"📊 Timeline segments: {len(timeline_segments)}")
            logger.info(f"🎨 Effects to apply: {[effect for seg in timeline_segments for effect in seg.effects]}")
            
            success = await renderer.render_video(
                video_path=temp_input_path,
                timeline=timeline,
                output_path=output_path,
                quality_preset=quality_enum
            )
            
            if not success:
                raise Exception("Multi-video rendering failed - check renderer logs for details")
            
            # Upload the rendered video to S3
            storage_key = f"custom_renders/{output_filename}"
            storage_url = await storage_client.upload_file(
                file_path=output_path,
                file_key=storage_key,
                content_type="video/mp4",
                metadata={
                    'project_id': str(project_id),
                    'video_ids': [str(vid) for vid in project.video_ids],
                    'render_timestamp': datetime.utcnow().isoformat(),
                    'effects_applied': [effect for seg in timeline_segments for effect in seg.effects],
                    'quality_preset': quality_preset
                }
            )
            
            # Create download URL
            download_url = await storage_client.create_download_url(storage_key)
            
            logger.info(f"✅ Multi-video custom effects video rendered and uploaded successfully")
            logger.info(f"📁 Output: {storage_key}")
            logger.info(f"🔗 Download URL: {download_url}")
            
            return JSONResponse({
                "success": True,
                "download_url": str(download_url),
                "filename": output_filename,
                "project_id": str(project_id),
                "message": "Multi-video project rendered with optimized shader effects",
                "effects_applied": [effect for seg in timeline_segments for effect in seg.effects]
            })
            
        finally:
            # Clean up temporary files
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"🧹 Temporary directory cleaned up: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Failed to clean up temp directory: {cleanup_error}")
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"❌ Multi-video custom effects rendering failed for project {project_id}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception message: {str(e)}")
        logger.error(f"❌ Full traceback:")
        logger.error(tb)
        raise HTTPException(status_code=500, detail=f"Multi-video custom effects rendering failed: {str(e)}")

@multi_video_router.get("/projects/{project_id}/download-rendered/{render_job_id}")
async def download_multi_video_rendered(
    project_id: UUID,
    render_job_id: str,
    storage_client=Depends(get_storage_client)
):
    """
    Download a rendered multi-video file by render job ID.
    This endpoint streams the file directly to avoid all CORS and SSL issues.
    """
    try:
        # Construct the S3 key for the rendered video
        output_key = f"custom_renders/multi_video_custom_effects_{project_id}_{render_job_id}.mp4"
        
        # Check if the file exists in S3
        try:
            storage_client.s3_client.head_object(Bucket=storage_client.bucket_name, Key=output_key)
        except Exception as e:
            logger.error(f"❌ Rendered multi-video not found: {output_key}")
            raise HTTPException(status_code=404, detail="Rendered multi-video not found")
        
        # Download the file from S3 to a temporary location
        import tempfile
        import os
        temp_file_path = os.path.join(tempfile.gettempdir(), f"download_multi_{render_job_id}.mp4")
        
        try:
            storage_client.s3_client.download_file(
                storage_client.bucket_name,
                output_key,
                temp_file_path
            )
            
            # Stream the file directly to the client
            from fastapi.responses import FileResponse
            return FileResponse(
                path=temp_file_path,
                media_type="video/mp4",
                filename=f"multi_video_custom_effects_{project_id}.mp4",
                background=lambda: os.remove(temp_file_path) if os.path.exists(temp_file_path) else None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to download multi-video file from S3: {e}")
            raise HTTPException(status_code=500, detail="Failed to download multi-video file from S3")
        
    except Exception as e:
        logger.error(f"❌ Failed to serve multi-video file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve multi-video file: {str(e)}")

@multi_video_router.post("/projects/{project_id}/test-auto-trigger", response_model=Dict[str, Any])
async def test_auto_trigger_editing_job(project_id: UUID):
    """Test endpoint to manually trigger the auto-triggering of editing jobs."""
    try:
        logger.info(f"[TEST AUTO-TRIGGER] Testing auto-trigger for project {project_id}")
        
        # Call the LLM recommendation function directly
        await _trigger_llm_recommendation_after_cross_analysis(str(project_id))
        
        # Get the updated project to see if editing job was created
        multi_video_manager = await get_multi_video_manager()
        project = await multi_video_manager.get_project(project_id)
        
        return {
            "success": True,
            "message": "Auto-trigger test completed",
            "project_id": str(project_id),
            "editing_job": str(project.editing_job) if project.editing_job else None,
            "cross_analysis_job": str(project.cross_analysis_job) if project.cross_analysis_job else None,
            "status": project.status
        }
        
    except Exception as e:
        logger.error(f"[TEST AUTO-TRIGGER] Test failed: {e}")
        import traceback
        logger.error(f"[TEST AUTO-TRIGGER] Full traceback: {traceback.format_exc()}")
        
        return {
            "success": False,
            "message": f"Auto-trigger test failed: {str(e)}",
            "project_id": str(project_id),
            "error": str(e)
        }

