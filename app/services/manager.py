"""
Service manager for graceful degradation and lazy loading.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from app.editor.renderer_simple import SimpleVideoRenderer

logger = logging.getLogger(__name__)


class MockRedis:
    """This class is deprecated. Use real Redis instead."""
    
    def __init__(self):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def enqueue(self, queue_name: str, job_data: Dict[str, Any]) -> str:
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def set_job_status(self, job_id: str, status: str, result: Optional[Dict] = None):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def create_analysis_job(self, video_id, template_type=None, analysis_options=None):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def create_editing_job(self, video_id, template_id=None, template_type=None, custom_settings=None, quality_preset=None):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def enqueue_job(self, job):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def get_job_by_video_id(self, video_id):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")
    
    async def list_jobs(self, page=1, page_size=10, status=None):
        raise NotImplementedError("MockRedis is deprecated. Please use real Redis.")


class MockTemplateManager:
    """This class is deprecated. Use real template manager instead."""
    
    def __init__(self):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def get_template(self, template_id):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def create_template(self, template):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def update_template(self, template_id, template):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def delete_template(self, template_id):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def list_templates(self, template_type=None):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")
    
    async def get_template_by_type(self, template_type):
        raise NotImplementedError("MockTemplateManager is deprecated. Please use real template manager.")


class MockStorage:
    """This class is deprecated. Use real storage instead."""
    
    def __init__(self):
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def upload_file(self, file_path: str, file_key: str, **kwargs) -> str:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def create_upload_url(self, filename: str, **kwargs) -> tuple:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def create_download_url(self, video_id_or_url, **kwargs) -> str:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def create_download_url_by_key(self, file_key: str, **kwargs) -> str:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def get_file_key(self, video_id: str, video_type: str = "processed") -> str:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")
    
    async def download_file(self, file_url: str, local_path: str) -> bool:
        raise NotImplementedError("MockStorage is deprecated. Please use real storage.")


class ServiceManager:
    """
    Service manager that provides lazy loading of real services.
    
    Services are initialized on-demand and require real implementations.
    """
    
    def __init__(self):
        self._redis = None
        self._storage = None
        self._analyzer = None
        self._renderer = None
        self._template_manager = None
        self._timeline_builder = None
        self._video_converter = None
        self._initialization_lock = asyncio.Lock()
    
    async def get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            async with self._initialization_lock:
                if self._redis is None:
                    try:
                        from app.job_queue.worker import initialize_redis_connection
                        from app.config.settings import get_settings
                        
                        settings = get_settings()
                        self._redis = await initialize_redis_connection(settings.redis_url)
                        logger.info("✅ Redis connection established")
                    except Exception as e:
                        logger.error(f"❌ Redis unavailable: {e}")
                        raise Exception(f"Redis connection failed: {e}")
        
        return self._redis
    
    async def get_storage(self):
        """Get storage client."""
        if self._storage is None:
            async with self._initialization_lock:
                if self._storage is None:
                    try:
                        from app.ingestion.storage import initialize_storage_client, get_storage_client
                        from app.config.settings import get_settings
                        
                        settings = get_settings()
                        
                        # Check if AWS credentials are configured
                        if settings.aws_access_key_id and settings.aws_secret_access_key:
                            # Use S3 storage if credentials are available
                            await initialize_storage_client(
                                aws_access_key_id=settings.aws_access_key_id,
                                aws_secret_access_key=settings.aws_secret_access_key,
                                aws_region=settings.aws_region,
                                bucket_name=settings.s3_bucket_name
                            )
                            self._storage = get_storage_client()
                            logger.info("✅ S3 storage client initialized")
                        else:
                            # Use local storage fallback for development
                            from app.services.local_storage import LocalStorage
                            self._storage = LocalStorage()
                            logger.info("✅ Local storage client initialized (development mode)")
                            
                    except Exception as e:
                        logger.error(f"❌ Storage unavailable: {e}")
                        # Try local storage as last resort
                        try:
                            from app.services.local_storage import LocalStorage
                            self._storage = LocalStorage()
                            logger.info("✅ Local storage client initialized (fallback mode)")
                        except Exception as fallback_error:
                            logger.error(f"❌ Local storage fallback also failed: {fallback_error}")
                            raise Exception(f"Storage connection failed: {e}")
        
        return self._storage
    
    async def get_analyzer(self):
        """Get analysis engine."""
        if self._analyzer is None:
            async with self._initialization_lock:
                if self._analyzer is None:
                    try:
                        from app.analyzer.engine import initialize_analysis_engine, get_analysis_engine
                        await initialize_analysis_engine()
                        self._analyzer = get_analysis_engine()
                        logger.info("✅ Analysis engine initialized")
                    except Exception as e:
                        logger.error(f"❌ Analysis engine unavailable: {e}")
                        raise Exception(f"Analysis engine initialization failed: {e}")
        
        return self._analyzer
    
    def get_renderer(self) -> SimpleVideoRenderer:
        """
        Get the universal video renderer that handles both single and multi-video projects.
        
        Returns:
            SimpleVideoRenderer: Universal renderer instance
        """
        if self._renderer is None:
            self._renderer = SimpleVideoRenderer()
            logger.info("✅ Universal SimpleVideoRenderer initialized")
        return self._renderer
    
    async def get_template_manager(self):
        """Get template manager."""
        if self._template_manager is None:
            async with self._initialization_lock:
                if self._template_manager is None:
                    try:
                        from app.templates.manager import initialize_template_manager, get_template_manager
                        await initialize_template_manager()
                        self._template_manager = get_template_manager()
                        logger.info("✅ Template manager initialized")
                    except Exception as e:
                        logger.error(f"❌ Template manager unavailable: {e}")
                        raise Exception(f"Template manager initialization failed: {e}")
        
        return self._template_manager
    
    async def get_timeline_builder(self):
        """Get timeline builder."""
        if self._timeline_builder is None:
            async with self._initialization_lock:
                if self._timeline_builder is None:
                    try:
                        from app.timeline.builder import initialize_timeline_builder, get_timeline_builder
                        await initialize_timeline_builder()
                        self._timeline_builder = get_timeline_builder()
                        logger.info("✅ Timeline builder initialized")
                    except Exception as e:
                        logger.error(f"❌ Timeline builder unavailable: {e}")
                        raise Exception(f"Timeline builder initialization failed: {e}")
        
        return self._timeline_builder
    
    async def get_video_converter(self):
        """Get video converter."""
        if self._video_converter is None:
            async with self._initialization_lock:
                if self._video_converter is None:
                    try:
                        from app.ingestion.video_converter import VideoConverter
                        self._video_converter = VideoConverter()
                        # Test FFmpeg availability
                        is_healthy = await self._video_converter.health_check()
                        if is_healthy:
                            logger.info("✅ Video converter initialized with FFmpeg")
                        else:
                            logger.error("❌ Video converter initialized but FFmpeg not available")
                            raise Exception("FFmpeg is required but not available")
                    except Exception as e:
                        logger.error(f"❌ Video converter unavailable: {e}")
                        raise Exception(f"Video converter initialization failed: {e}")
        
        return self._video_converter
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all services."""
        health_status = {
            "core": "healthy",
            "redis": "unknown",
            "storage": "unknown",
            "analyzer": "unknown",
            "renderer": "unknown",
            "template_manager": "unknown",
            "timeline_builder": "unknown",
            "video_converter": "unknown"
        }
        
        # Check Redis
        try:
            redis = await self.get_redis()
            health_status["redis"] = "healthy"
        except Exception:
            health_status["redis"] = "unhealthy"
        
        # Check Storage
        try:
            storage = await self.get_storage()
            health_status["storage"] = "healthy"
        except Exception:
            health_status["storage"] = "unhealthy"
        
        # Check other services
        for service_name, getter in [
            ("analyzer", self.get_analyzer),
            ("renderer", self.get_renderer),
            ("template_manager", self.get_template_manager),
            ("timeline_builder", self.get_timeline_builder),
            ("video_converter", self.get_video_converter)
        ]:
            try:
                service = await getter()
                health_status[service_name] = "healthy"
            except Exception:
                health_status[service_name] = "unhealthy"
        
        return health_status


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


@lru_cache()
def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager 