"""
Main application entry point for the AI-Powered Video Editing Automation Engine.

This module orchestrates all components of the video editing pipeline:
- FastAPI application setup
- Background job queue initialization
- Module integration and dependency injection
- Health checks and monitoring endpoints

Deployed: Full backend with all API endpoints - Railway deployment - FORCE REDEPLOY
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Import our modules
from app.config.settings import get_settings
from app.api.routes import health_router, video_router, template_router, multi_video_router
from app.api.auth import router as auth_router
from app.api.music_routes import router as music_router
from app.ingestion.storage import initialize_storage_client
from app.analyzer.engine import initialize_analysis_engine
from app.editor.renderer_simple import initialize_renderer
from app.templates.manager import initialize_template_manager
from app.timeline.builder import initialize_timeline_builder
from app.services.redis_validator import RedisValidator
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")  # Also log to file for debugging
    ]
)

# Set specific log levels for verbose components
logging.getLogger("moviepy").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Uses service manager for non-blocking startup with graceful degradation.
    """
    logger.info("🚀 Starting Video Editing Automation Engine...")
    
    # Initialize configuration
    settings = get_settings()
    logger.info(f"Configuration loaded for environment: {settings.environment}")
    
    # Initialize service manager (non-blocking)
    from app.services import get_service_manager
    service_manager = get_service_manager()
    
    # Start services initialization in background
    async def initialize_services():
        try:
            logger.info("🔄 Initializing services in background...")
            
            # Initialize services lazily (they'll be ready when needed)
            await service_manager.get_redis()
            await service_manager.get_storage()
            await service_manager.get_analyzer()
            service_manager.get_renderer()  # Not async
            await service_manager.get_template_manager()
            await service_manager.get_timeline_builder()
            
            logger.info("🎬 Video Editing Automation Engine services ready!")
            
        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise Exception(f"Service initialization failed: {e}")
    
    # Start services initialization in background
    asyncio.create_task(initialize_services())
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Video Editing Automation Engine...")
    # Add cleanup logic here (close connections, etc.)


# Custom CORS middleware since the built-in one isn't working
class CustomCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"🔧 [CORS DEBUG] Middleware executing for {request.method} {request.url}")
        print(f"🔧 [CORS DEBUG] Origin header: {request.headers.get('origin')}")
        
        response = await call_next(request)
        
        # Add CORS headers manually
        origin = request.headers.get("origin")
        print(f"🔧 [CORS DEBUG] Processing origin: {origin}")
        
        if origin and origin in [
            "http://localhost:3001",
            "http://localhost:3000", 
            "https://localhost:3000",
            "https://localhost:8443",
            "http://localhost:8000",
            "http://127.0.0.1:3001",
            "https://127.0.0.1:3000",
            "https://127.0.0.1:8443",
            "https://frontend-clean-production.up.railway.app"
        ]:
            print(f"🔧 [CORS DEBUG] Adding CORS headers for origin: {origin}")
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
            response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-API-Key, Cache-Control, Pragma, Origin, Referer, User-Agent, DNT, Connection, Upgrade-Insecure-Requests"
            response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
            response.headers["Access-Control-Max-Age"] = "86400"
            print(f"🔧 [CORS DEBUG] CORS headers added: {dict(response.headers)}")
        else:
            print(f"🔧 [CORS DEBUG] Origin not in allowed list: {origin}")
        
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            print(f"🔧 [CORS DEBUG] Handling OPTIONS preflight request")
            response = JSONResponse(content={}, status_code=200)
            if origin and origin in [
                "http://localhost:3001",
                "http://localhost:3000", 
                "https://localhost:3000",
                "https://localhost:8443",
                "http://localhost:8000",
                "http://127.0.0.1:3001",
                "https://127.0.0.1:3000",
                "https://127.0.0.1:8443"
            ]:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
                response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-API-Key, Cache-Control, Pragma, Origin, Referer, User-Agent, DNT, Connection, Upgrade-Insecure-Requests"
                response.headers["Access-Control-Max-Age"] = "86400"
                print(f"🔧 [CORS DEBUG] OPTIONS CORS headers added: {dict(response.headers)}")
        
        return response


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application with all routes and middleware.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="AI-Powered Video Editing Automation Engine",
        description="Automated video editing with intelligent analysis and template-based rendering",
        version="1.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware - Use environment configuration
    cors_origins = settings.allowed_origins
    print(f"🔧 CORS origins loaded: {cors_origins}")
    
    # Remove broken built-in CORS middleware - using custom one instead
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=cors_origins,  # Use the loaded CORS origins from settings
    #     allow_credentials=True,
    #     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    #     allow_headers=[
    #         "Accept",
    #         "Accept-Language", 
    #         "Content-Language",
    #         "Content-Type",
    #         "Authorization",
    #         "X-Requested-With",
    #         "X-API-Key",
    #         "Cache-Control",
    #         "Pragma",
    #         "Origin",
    #         "Referer",
    #         "User-Agent",
    #         "DNT",
    #         "Connection",
    #         "Upgrade-Insecure-Requests",
    #     ],
    #     expose_headers=["Content-Length", "Content-Type"],
    #     max_age=86400,  # Cache preflight for 24 hours
    # )

    # Add custom CORS middleware since the built-in one is broken
    app.add_middleware(CustomCORSMiddleware)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint for basic health check."""
        return {
            "status": "healthy",
            "message": "Video Editing Automation Engine is running - FULL BACKEND DEPLOYED",
            "version": "1.0.0",
            "docs": "/docs",
            "deployed_at": "2024-01-01T00:00:00Z",
            "deployment_trigger": "manual_redeploy_$(date +%s)"
        }
    
    # Include API routes
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(video_router, prefix="/api/v1/videos", tags=["Videos"])
    app.include_router(template_router, prefix="/api/v1/templates", tags=["Templates"])
    app.include_router(multi_video_router, prefix="/api/v1", tags=["Multi-Video Projects"])
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(music_router, tags=["Music Library"])
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        import traceback
        logger.error(f"❌ [GLOBAL ERROR] Unhandled exception in {request.method} {request.url}")
        logger.error(f"❌ [GLOBAL ERROR] Exception type: {type(exc).__name__}")
        logger.error(f"❌ [GLOBAL ERROR] Exception message: {str(exc)}")
        logger.error(f"❌ [GLOBAL ERROR] Full traceback:")
        logger.error(traceback.format_exc())
        
        # Log request details for debugging
        logger.error(f"❌ [GLOBAL ERROR] Request details:")
        logger.error(f"   - Method: {request.method}")
        logger.error(f"   - URL: {request.url}")
        logger.error(f"   - Headers: {dict(request.headers)}")
        logger.error(f"   - Client: {request.client}")
        
        # Check if this is a validation error
        if "validation" in str(exc).lower() or "pydantic" in str(exc).lower():
            return JSONResponse(
                status_code=422,
                content={
                    "detail": f"Validation error: {str(exc)}",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)
                }
            )
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {str(exc)}",
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            }
        )
    
    return app


# Create the FastAPI app instance at module level for uvicorn
app = create_app()


def main():
    """
    Main entry point for the application.
    
    Starts the Uvicorn server with the configured settings.
    """
    settings = get_settings()
    
    # Use Railway's PORT environment variable if available
    port = int(os.environ.get("PORT", settings.port))
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main() 