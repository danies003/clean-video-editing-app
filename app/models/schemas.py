"""
Pydantic schemas for video editing automation engine.

This module defines all data models used throughout the application,
including request/response DTOs, analysis results, and configuration schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl


class ProcessingStatus(str, Enum):
    """Status enumeration for video processing jobs."""
    PENDING = "pending"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    EDITING = "editing"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WMV = "wmv"


class TemplateType(str, Enum):
    """Available template types for video editing."""
    BEAT_MATCH = "beat_match"
    CINEMATIC = "cinematic"
    FAST_PACED = "fast_paced"
    SLOW_MOTION = "slow_motion"
    TRANSITION_HEAVY = "transition_heavy"
    MINIMAL = "minimal"


class QualityPreset(str, Enum):
    """Video quality presets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class EditStyle(str, Enum):
    """Available editing styles."""
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    CINEMATIC = "cinematic"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    details: Optional[Dict[str, Any]] = None


# Video Upload Models
class VideoUploadRequest(BaseModel):
    """Request model for video upload."""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the video")
    file_size: int = Field(..., description="File size in bytes")
    template_type: Optional[TemplateType] = Field(default=TemplateType.BEAT_MATCH)
    quality_preset: QualityPreset = Field(default=QualityPreset.HIGH)
    custom_settings: Optional[Dict[str, Any]] = Field(default=None)
    video_url: Optional[HttpUrl] = Field(default=None, description="URL to an existing video file to use instead of uploading")


class VideoUploadResponse(BaseResponse):
    """Response model for video upload."""
    video_id: str
    upload_url: HttpUrl
    expires_at: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Video metadata")


# Video Analysis Models
class BeatDetectionResult(BaseModel):
    """Result of beat detection analysis."""
    timestamps: List[float] = Field(..., description="Beat timestamps in seconds")
    confidence_scores: List[float] = Field(..., description="Confidence scores for each beat")
    bpm: float = Field(..., description="Estimated beats per minute")
    energy_levels: List[float] = Field(..., description="Energy levels at each beat")


class MotionAnalysisResult(BaseModel):
    """Result of motion analysis."""
    motion_spikes: List[float] = Field(..., description="Timestamps of motion spikes")
    motion_intensities: List[float] = Field(..., description="Intensity of motion at each spike")
    scene_changes: List[float] = Field(..., description="Timestamps of scene changes")
    scene_confidence: List[float] = Field(..., description="Confidence scores for scene changes")
    motion_score: float = Field(default=0.0, description="Overall motion intensity score")
    activity_level: str = Field(default="low", description="Activity level (low, medium, high)")


class AudioAnalysisResult(BaseModel):
    """Result of audio analysis."""
    volume_levels: List[float] = Field(..., description="Volume levels over time")
    silence_periods: List[List[float]] = Field(..., description="Silence period timestamps (start, end) as lists")
    audio_peaks: List[float] = Field(..., description="Audio peak timestamps")
    frequency_analysis: Dict[str, List[float]] = Field(..., description="Frequency domain analysis")


class VideoAnalysisResult(BaseModel):
    """Complete video analysis result."""
    video_id: UUID
    duration: float = Field(..., description="Video duration in seconds")
    fps: float = Field(..., description="Video frame rate")
    resolution: tuple = Field(..., description="Video resolution (width, height)")
    beat_detection: BeatDetectionResult
    motion_analysis: MotionAnalysisResult
    audio_analysis: AudioAnalysisResult
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


# Template Models
class EditingTemplate(BaseModel):
    """Template for video editing rules."""
    template_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Template name")
    template_type: TemplateType
    description: str = Field(..., description="Template description")
    
    # Editing rules
    transition_duration: float = Field(default=0.5, description="Default transition duration")
    cut_sensitivity: float = Field(default=0.7, description="Sensitivity for automatic cuts")
    beat_sync_threshold: float = Field(default=0.1, description="Threshold for beat synchronization")
    
    # Visual effects
    effects: List[str] = Field(default_factory=list, description="List of effects to apply")
    color_grading: Optional[Dict[str, Any]] = Field(default=None, description="Color grading settings")
    
    # Audio settings
    audio_fade_in: float = Field(default=0.0, description="Audio fade in duration")
    audio_fade_out: float = Field(default=0.0, description="Audio fade out duration")
    
    # Rendering settings
    output_format: VideoFormat = Field(default=VideoFormat.MP4)
    quality_preset: QualityPreset = Field(default=QualityPreset.HIGH)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TemplateListResponse(BaseResponse):
    """Response model for template listing."""
    templates: List[EditingTemplate]
    total_count: int


# Timeline Models
class TimelineSegment(BaseModel):
    """A segment in the video timeline."""
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    source_video_id: UUID
    effects: List[str] = Field(default_factory=list)
    transition_in: Optional[str] = Field(default=None, description="Transition effect at start")
    transition_out: Optional[str] = Field(default=None, description="Transition effect at end")
    effectCustomizations: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Effect customizations with parameters")
    # Audio timing fields for multi-video projects
    audio_start_time: Optional[float] = Field(default=None, description="Audio start time in sequential timeline")
    audio_end_time: Optional[float] = Field(default=None, description="Audio end time in sequential timeline")

    # LLM-driven ordering fields (optional for backward compatibility)
    segment_order: Optional[int] = Field(default=None, description="Order in final sequence (0, 1, 2, ...)")
    llm_reasoning: Optional[str] = Field(default=None, description="Why LLM chose this segment")
    confidence_score: Optional[float] = Field(default=None, description="LLM confidence in this segment")
    segment_tags: Optional[List[str]] = Field(default=None, description="LLM-assigned tags")


class VideoTimeline(BaseModel):
    """Complete video timeline for editing."""
    timeline_id: UUID = Field(default_factory=uuid4)
    video_id: UUID
    template: EditingTemplate
    segments: List[TimelineSegment] = Field(default_factory=list)
    total_duration: float = Field(..., description="Total timeline duration")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Job Models
class ProcessingJob(BaseModel):
    """Background processing job model."""
    job_id: UUID = Field(default_factory=uuid4)
    video_id: UUID
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    progress: float = Field(default=0.0, description="Progress percentage (0-100)")
    
    # Job details
    template_type: TemplateType
    quality_preset: QualityPreset
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    analysis_result: Optional[VideoAnalysisResult] = None
    timeline: Optional[VideoTimeline] = None
    output_url: Optional[HttpUrl] = None
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseResponse):
    """Response model for job status."""
    job: ProcessingJob


class JobListResponse(BaseResponse):
    """Response model for job listing."""
    jobs: List[ProcessingJob]
    total_count: int
    page: int
    page_size: int


# API Request Models
class AnalyzeVideoRequest(BaseModel):
    """Request model for video analysis."""
    video_url: HttpUrl = Field(..., description="URL to the video file to analyze")
    template_type: Optional[TemplateType] = Field(default=TemplateType.BEAT_MATCH)
    analysis_options: Optional[Dict[str, Any]] = Field(default=None)


class EditVideoRequest(BaseModel):
    """Request model for video editing."""
    template_id: Optional[UUID] = Field(default=None)
    template_type: TemplateType = Field(default=TemplateType.BEAT_MATCH)
    custom_settings: Optional[Dict[str, Any]] = Field(default=None)
    quality_preset: QualityPreset = Field(default=QualityPreset.HIGH)


class AdvancedEditRequest(BaseModel):
    video_id: Optional[UUID] = None  # Optional since it's in the URL path
    edit_scale: float = 0.5  # 0.0 (minimal) to 1.0 (maximal)
    style_preferences: Dict[str, str] = Field(default_factory=dict)
    target_duration: Optional[float] = None
    dry_run: Optional[bool] = False

class EditDecisionSegment(BaseModel):
    start: float
    end: float
    transition: Optional[str] = None
    transition_duration: Optional[float] = 0.5
    tags: List[str] = []
    speed: Optional[float] = 1.0
    effectCustomizations: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Effect customizations with parameters")

class EditDecisionMap(BaseModel):
    video_id: UUID
    style: str
    segments: List[EditDecisionSegment]
    notes: Optional[str] = None
    edit_scale: float = 0.5


class SegmentRecommendations(BaseModel):
    """AI recommendations for video segments."""
    segment_reasoning: str = Field(..., description="Reasoning for segment selection")
    transition_reasoning: str = Field(..., description="Reasoning for transition choice")
    effects_reasoning: str = Field(..., description="Reasoning for effects application")
    arrangement_reasoning: str = Field(..., description="Reasoning for segment arrangement")
    confidence_score: float = Field(..., description="AI confidence in recommendations")
    alternative_suggestions: List[str] = Field(default_factory=list, description="Alternative editing suggestions")


class MoviePyEffect(BaseModel):
    """MoviePy effect configuration"""
    effect_type: str = Field(..., description="Type of effect to apply")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")
    start_time: Optional[float] = Field(default=None, description="Effect start time (relative to segment)")
    end_time: Optional[float] = Field(default=None, description="Effect end time (relative to segment)")
    intensity: float = Field(default=1.0, description="Effect intensity (0.0 to 1.0)")


class MoviePyTransition(BaseModel):
    """MoviePy transition configuration"""
    transition_type: str = Field(..., description="Type of transition")
    duration: float = Field(default=0.5, description="Transition duration in seconds")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transition parameters")
    easing: str = Field(default="linear", description="Easing function (linear, ease_in, ease_out, ease_in_out)")


class MoviePySegment(BaseModel):
    """MoviePy-compatible video segment"""
    start_time: float = Field(..., description="Start time in source video (seconds)")
    end_time: float = Field(..., description="End time in source video (seconds)")
    speed: float = Field(default=1.0, description="Playback speed multiplier")
    volume: float = Field(default=1.0, description="Audio volume (0.0 to 1.0)")
    effects: List[MoviePyEffect] = Field(default_factory=list, description="Visual effects to apply")
    audio_effects: List[MoviePyEffect] = Field(default_factory=list, description="Audio effects to apply")
    transition_in: Optional[MoviePyTransition] = Field(default=None, description="Incoming transition")
    transition_out: Optional[MoviePyTransition] = Field(default=None, description="Outgoing transition")
    crop: Optional[Dict[str, float]] = Field(default=None, description="Crop parameters (x1, y1, x2, y2)")
    resize: Optional[Dict[str, int]] = Field(default=None, description="Resize parameters (width, height)")
    rotation: float = Field(default=0.0, description="Rotation in degrees")
    brightness: float = Field(default=1.0, description="Brightness multiplier")
    contrast: float = Field(default=1.0, description="Contrast multiplier")
    saturation: float = Field(default=1.0, description="Saturation multiplier")
    gamma: float = Field(default=1.0, description="Gamma correction")
    blur: float = Field(default=0.0, description="Blur radius")
    sharpness: float = Field(default=1.0, description="Sharpness multiplier")


class MoviePyRenderingPlan(BaseModel):
    """Complete MoviePy rendering plan"""
    video_id: UUID
    source_video_path: str = Field(..., description="Path to source video file")
    output_path: str = Field(..., description="Path for output video file")
    target_duration: float = Field(..., description="Target duration in seconds")
    output_format: str = Field(default="mp4", description="Output video format")
    output_quality: str = Field(default="high", description="Output quality preset")
    fps: float = Field(default=30.0, description="Output frame rate")
    resolution: Optional[tuple] = Field(default=None, description="Output resolution (width, height)")
    
    # Rendering segments
    segments: List[MoviePySegment] = Field(..., description="Video segments to render")
    
    # Global settings
    global_effects: List[MoviePyEffect] = Field(default_factory=list, description="Global effects applied to entire video")
    audio_settings: Dict[str, Any] = Field(default_factory=dict, description="Global audio settings")
    color_settings: Dict[str, Any] = Field(default_factory=dict, description="Global color settings")
    
    # Metadata
    style: str = Field(..., description="Editing style used")
    confidence: float = Field(..., description="AI confidence in the plan")
    reasoning: str = Field(..., description="AI reasoning for the editing decisions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Available effects and transitions for reference
    available_effects: List[str] = Field(default_factory=lambda: [
        "fade_in", "fade_out", "crossfade", "slide_in", "slide_out", "zoom_in", "zoom_out",
        "rotate", "flip", "mirror", "crop", "resize", "brightness", "contrast", "saturation",
        "gamma", "blur", "sharpen", "color_balance", "sepia", "black_white", "vintage",
        "speed_up", "slow_motion", "freeze_frame", "loop", "reverse", "audio_normalize",
        "audio_fade_in", "audio_fade_out", "audio_speed", "audio_volume", "audio_echo",
        "audio_reverb", "audio_high_pass", "audio_low_pass", "whip_pan", "zoom_blur",
        "shake", "match_cut", "dissolve", "wipe", "iris", "page_turn", "morph"
    ])
    
    available_transitions: List[str] = Field(default_factory=lambda: [
        "fade", "crossfade", "dissolve", "wipe", "slide", "zoom", "rotate", "flip",
        "iris", "page_turn", "morph", "whip_pan", "zoom_blur", "shake", "match_cut",
        "push", "pull", "split", "stretch", "squeeze", "bounce", "elastic", "spring"
    ])


# Multi-Video Project Models
class MultiVideoProject(BaseModel):
    """Multi-video project model."""
    project_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Project name")
    video_ids: List[UUID] = Field(..., description="List of video IDs in the project")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    progress: float = Field(default=0.0, description="Project progress percentage (0-100)")
    analysis_jobs: List[UUID] = Field(default_factory=list, description="Individual analysis job IDs")
    cross_analysis_job: Optional[UUID] = Field(default=None, description="Cross-video analysis job ID")
    editing_job: Optional[UUID] = Field(default=None, description="Multi-video editing job ID")
    output_video_id: Optional[UUID] = Field(default=None, description="Output video ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossVideoSettings(BaseModel):
    """Settings for cross-video analysis."""
    enable_cross_analysis: bool = Field(default=True, description="Enable cross-video analysis")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for cross-video matching")
    chunking_strategy: str = Field(default="scene", description="Chunking strategy (scene, action, audio, content)")


class MultiVideoUploadRequest(BaseModel):
    """Request model for multi-video project creation."""
    project_name: str = Field(..., description="Name of the multi-video project")
    cross_video_settings: CrossVideoSettings = Field(default_factory=CrossVideoSettings)


class MultiVideoUploadResponse(BaseResponse):
    """Response model for multi-video project creation."""
    project_id: UUID
    video_ids: List[UUID]
    upload_urls: List[HttpUrl]
    expires_at: str
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class MultiVideoAnalysisRequest(BaseModel):
    """Request model for multi-video analysis."""
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for cross-video matching")
    chunking_strategy: str = Field(default="scene", description="Chunking strategy")
    cross_analysis_settings: Dict[str, Any] = Field(default_factory=dict)


class CrossVideoAnalysisRequest(BaseModel):
    """Request model for cross-video analysis."""
    project_id: UUID = Field(..., description="Project ID")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for cross-video matching")
    chunking_strategy: str = Field(default="scene", description="Chunking strategy")
    cross_analysis_settings: Dict[str, Any] = Field(default_factory=dict)


class CrossVideoAnalysisResult(BaseModel):
    """Result of cross-video analysis."""
    project_id: UUID
    analysis_job_id: UUID
    cross_analysis_job_id: Optional[UUID] = None
    similarity_matrix: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    cross_video_segments: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiVideoEditRequest(BaseModel):
    """Request model for multi-video editing."""
    edit_scale: float = Field(default=0.7, description="Edit intensity (0.0 to 1.0)")
    style_preferences: Dict[str, str] = Field(default_factory=dict)
    cross_video_effects: Dict[str, Any] = Field(default_factory=dict)
    target_duration: Optional[float] = Field(default=None, description="Target duration in seconds")


class MultiVideoEditResponse(BaseResponse):
    """Response model for multi-video editing."""
    project_id: UUID
    editing_job_id: UUID
    estimated_duration: Optional[float] = Field(default=None, description="Estimated processing time")


class MultiVideoProjectStatus(BaseModel):
    """Status model for multi-video projects."""
    project_id: UUID
    status: ProcessingStatus
    video_ids: List[UUID] = Field(default_factory=list, description="List of video IDs in the project")
    analysis_completed: int = Field(default=0, description="Number of completed analysis jobs")
    cross_analysis_job: Optional[UUID] = Field(default=None, description="Cross-analysis job ID if exists")
    cross_analysis_completed: bool = Field(default=False, description="Cross-analysis completion status")
    editing_completed: bool = Field(default=False, description="Editing completion status")
    progress: float = Field(default=0.0, description="Overall progress percentage")
    output_video_url: Optional[HttpUrl] = Field(default=None, description="Output video URL")
    error: Optional[str] = Field(default=None, description="Error message if any")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Project metadata including LLM plan")


# Health Check Models
class HealthCheckResponse(BaseResponse):
    """Health check response model."""
    status: str = "healthy"
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
    uptime: float = Field(..., description="Application uptime in seconds")


# Export all models
__all__ = [
    "ProcessingStatus",
    "VideoFormat", 
    "TemplateType",
    "QualityPreset",
    "BaseResponse",
    "ErrorResponse",
    "VideoUploadRequest",
    "VideoUploadResponse",
    "BeatDetectionResult",
    "MotionAnalysisResult", 
    "AudioAnalysisResult",
    "VideoAnalysisResult",
    "EditingTemplate",
    "TemplateListResponse",
    "TimelineSegment",
    "VideoTimeline",
    "ProcessingJob",
    "JobStatusResponse",
    "JobListResponse",
    "AnalyzeVideoRequest",
    "EditVideoRequest",
    "HealthCheckResponse"
] 