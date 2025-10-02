"""
Simple backend for testing deployment
"""
import logging
import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Editor API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the existing auth routes
try:
    from app.api.auth import router as auth_router
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    logger.info("✅ Auth routes loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Could not load auth routes: {e}")

@app.get("/")
async def root():
    return {"message": "AI Video Editor API is running!"}

@app.get("/health")
async def health():
    logger.info("Health check endpoint called")
    return {
        "status": "healthy", 
        "service": "ai-video-editor",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

