"""
Application settings and configuration management.

This module handles all configuration settings for the video editing automation engine,
including environment variables, database connections, and service endpoints.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import traceback
import logging

# Set up logging for settings validation
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Provides type-safe configuration management with environment-specific defaults.
    """
    
    # Application Settings
    app_name: str = "Video Editing Automation Engine"
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS Settings - Allow both HTTP and HTTPS, but prioritize specific origins for security
    allowed_origins: List[str] = [
        "http://localhost:3001",      # Frontend HTTP (development)
        "http://localhost:3000",      # Frontend HTTP (alternative)
        "https://localhost:3000",     # Frontend HTTPS (proxy)
        "https://localhost:8443",     # Backend HTTPS (proxy)
        "http://localhost:8000",      # Backend HTTP (direct)
        "http://127.0.0.1:3001",     # Frontend HTTP (127.0.0.1)
        "https://127.0.0.1:3000",    # Frontend HTTPS (127.0.0.1)
        "https://127.0.0.1:8443",    # Backend HTTPS (127.0.0.1)
    ]
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed_origins from environment variable, handling string representations."""
        try:
            if isinstance(v, str):
                # Handle string representation of list from .env file
                if v.startswith('[') and v.endswith(']'):
                    try:
                        # Remove brackets and split by comma, then clean up quotes and spaces
                        content = v[1:-1]
                        items = [item.strip().strip('"\'') for item in content.split(',')]
                        logger.info(f"✅ Successfully parsed allowed_origins from string: {items}")
                        return items
                    except Exception as parse_error:
                        logger.error(f"❌ Failed to parse allowed_origins string '{v}': {parse_error}")
                        # Return safe defaults
                        return ["http://localhost:3001", "http://localhost:3000", "https://localhost:3000"]
                # If it's a single URL string, wrap it in a list
                logger.info(f"✅ Single URL string converted to list: [{v}]")
                return [v]
            elif isinstance(v, list):
                logger.info(f"✅ allowed_origins already a list: {v}")
                return v
            else:
                logger.warning(f"⚠️ Unexpected allowed_origins type {type(v)}, using defaults")
                return ["http://localhost:3001", "http://localhost:3000", "https://localhost:3000"]
        except Exception as e:
            logger.error(f"❌ Critical error parsing allowed_origins: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Always return safe defaults to prevent system failure
            return ["http://localhost:3001", "http://localhost:3000", "https://localhost:3000"]
    
    # API Authentication
    api_key: str = ""
    api_key_header: str = "X-API-Key"
    
    # Redis Settings
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # AWS S3 Settings
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""
    s3_endpoint_url: Optional[str] = None
    
    # Video Processing Settings
    max_video_size_mb: int = 500
    supported_video_formats: List[str] = ["mp4", "avi", "mov", "mkv", "wmv"]
    temp_directory: str = "/tmp/video_editing"
    
    # Analysis Settings
    audio_sample_rate: int = 22050
    video_fps: int = 30
    motion_threshold: float = 0.1
    beat_detection_sensitivity: float = 0.5
    use_real_analysis: bool = True  # Set to False to use mock analysis only
    
    # Rendering Settings
    output_format: str = "mp4"
    output_quality: str = "high"
    max_rendering_time_minutes: int = 30
    
    # Job Queue Settings
    queue_name: str = "video_editing"
    max_workers: int = 4
    job_timeout: int = 3600  # 1 hour
    
    # Gemini AI Settings
    gemini_api_key: str = ""
    
    # OAuth Settings
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    facebook_app_id: Optional[str] = None
    facebook_app_secret: Optional[str] = None
    # Optional: allow using a prebuilt app token (app_id|app_secret) for debug_token in dev
    facebook_app_token: Optional[str] = None
    
    # JWT Settings
    jwt_secret_key: str = "your_jwt_secret_key_here"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Video Resolution Settings
    target_video_width: int = 1080
    target_video_height: int = 1920
    standardize_video_resolution: bool = True
    disable_video_resize: bool = False  # Set to True to prevent segmentation faults on macOS
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields from environment


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration instance
        
    Note:
        This function is cached to avoid reloading settings on every call.
        Settings are only reloaded when the application restarts.
    """
    return Settings()


# Export settings instance for easy access
settings = get_settings() 