"""
Railway-optimized main application entry point for the AI-Powered Video Editing Automation Engine.
This version is optimized for Railway deployment with faster startup and better error handling.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Custom CORS middleware for Railway
class RailwayCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add CORS headers for Railway
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, X-API-Key, Cache-Control, Pragma, Origin, Referer, User-Agent, DNT, Connection, Upgrade-Insecure-Requests"
        
        return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified lifespan manager for Railway deployment."""
    logger.info("🚀 Starting Video Editing Automation Engine on Railway - DEPLOYMENT v2.1...")
    yield
    logger.info("🛑 Shutting down Video Editing Automation Engine...")

def create_railway_app() -> FastAPI:
    """Create Railway-optimized FastAPI application."""
    
    app = FastAPI(
        title="AI-Powered Video Editing Automation Engine",
        description="Railway-optimized version with full API endpoints",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(RailwayCORSMiddleware)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "status": "healthy",
            "message": "Video Editing Automation Engine is running - RAILWAY DEPLOYMENT",
            "version": "1.0.0",
            "docs": "/docs",
            "deployed_at": "2024-01-01T00:00:00Z"
        }
    
    # Health endpoints for Railway
    @app.get("/health/")
    async def health():
        return {
            "status": "healthy",
            "service": "ai-video-editor",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "1.0.0"
        }
    
    @app.get("/health/simple")
    async def health_simple():
        return {"status": "healthy"}
    
    # Load full backend routes - NO FALLBACK
    logger.info("🔄 Loading full backend routes...")
    
    try:
        logger.info("🔄 Step 1: Importing health_router...")
        from app.api.routes import health_router
        logger.info("✅ health_router imported successfully")
        
        logger.info("🔄 Step 2: Importing video_router...")
        from app.api.routes import video_router
        logger.info("✅ video_router imported successfully")
        
        logger.info("🔄 Step 3: Importing template_router...")
        from app.api.routes import template_router
        logger.info("✅ template_router imported successfully")
        
        logger.info("🔄 Step 4: Importing multi_video_router...")
        from app.api.routes import multi_video_router
        logger.info("✅ multi_video_router imported successfully")
        
        logger.info("🔄 Step 5: Importing auth_router...")
        from app.api.auth import router as auth_router
        logger.info("✅ auth_router imported successfully")
        
        logger.info("🔄 Step 6: Importing music_router...")
        from app.api.music_routes import router as music_router
        logger.info("✅ music_router imported successfully")
        
        logger.info("🔄 Step 7: Including routers...")
        # Include API routes
        app.include_router(health_router, prefix="/health", tags=["Health"])
        app.include_router(video_router, prefix="/api/v1/videos", tags=["Videos"])
        app.include_router(template_router, prefix="/api/v1/templates", tags=["Templates"])
        app.include_router(multi_video_router, prefix="/api/v1", tags=["Multi-Video Projects"])
        app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
        app.include_router(music_router, tags=["Music Library"])
        
        logger.info("✅ Full backend routes loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load full backend routes: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise e  # Re-raise the error to fail the deployment
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"❌ [GLOBAL ERROR] {request.method} {request.url}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal server error: {str(exc)}",
                "error_type": type(exc).__name__
            }
        )
    
    return app

# Create the FastAPI app instance
app = create_railway_app()

def main():
    """Main entry point optimized for Railway."""
    # Use Railway's PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"🚀 Starting Railway-optimized server on {host}:{port}")
    
    uvicorn.run(
        "main_railway:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
# Force deployment test - Thu Oct  9 20:16:58 EDT 2025
