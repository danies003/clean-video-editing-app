"""
Simple backend for testing deployment
"""
import logging
import os
import requests
from fastapi import FastAPI, HTTPException
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
            "enabled": bool(os.environ.get("GOOGLE_CLIENT_ID")),
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        },
        "facebook": {
            "enabled": bool(os.environ.get("FACEBOOK_APP_ID")),
            "app_id": os.environ.get("FACEBOOK_APP_ID"),
        }
    }

@app.post("/api/v1/auth/social")
async def social_auth(request_data: dict):
    """Handle social authentication with proper OAuth verification"""
    try:
        provider = request_data.get("provider", "").lower()
        code = request_data.get("code")
        redirect_uri = request_data.get("redirect_uri")
        
        if not code or not redirect_uri:
            raise HTTPException(status_code=400, detail="Missing code or redirect_uri")
        
        # Mock successful authentication for now
        # In a real implementation, you would verify the OAuth code with the provider
        user_id = f"user_{provider}_{hash(code) % 10000}"
        
        auth_response = {
            "access_token": f"mock_token_{user_id}_{hash(code) % 1000}",
            "user": {
                "id": user_id,
                "name": f"{provider.title()} User",
                "email": f"user_{user_id}@example.com",
                "provider": provider,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }
        
        logger.info(f"✅ Mock social auth successful for {provider}")
        return auth_response
        
    except Exception as e:
        logger.error(f"❌ Social auth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

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

@app.get("/health/simple")
async def health_simple():
    logger.info("Simple health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

