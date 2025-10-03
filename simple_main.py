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

# Include basic auth routes without video processing dependencies
@app.get("/api/v1/auth/config")
async def auth_config():
    """Get authentication configuration"""
    return {
        "google": {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        },
        "facebook": {
            "app_id": os.environ.get("FACEBOOK_APP_ID"),
        }
    }

@app.post("/api/v1/auth/social")
async def social_auth():
    """Handle social authentication"""
    return {"message": "Social auth endpoint - needs proper implementation"}

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

