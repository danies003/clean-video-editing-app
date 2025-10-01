"""
Authentication API routes for user registration, login, and social authentication.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from app.models.auth import UserSignUp, UserSignIn, SocialAuth, UserResponse, AuthResponse
from app.services.auth_service import AuthService
from app.services.oauth_service import oauth_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["authentication"])
security = HTTPBearer()

def get_auth_service() -> AuthService:
    return AuthService()

@router.post("/signup", response_model=AuthResponse)
async def signup(
    user_data: UserSignUp,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user"""
    try:
        return auth_service.register_user(user_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/signin", response_model=AuthResponse)
async def signin(
    user_data: UserSignIn,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Sign in with email and password"""
    try:
        return auth_service.authenticate_user(user_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sign in failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sign in failed: {str(e)}"
        )

@router.post("/social", response_model=AuthResponse)
async def social_auth(
    social_data: SocialAuth,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate with social provider (Google/Facebook)"""
    try:
        return auth_service.authenticate_social_user(social_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Social authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Social authentication failed: {str(e)}"
        )

@router.get("/config")
async def get_oauth_config():
    """Get OAuth configuration for frontend"""
    return oauth_service.get_oauth_config()

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: str = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get current user information"""
    user = auth_service.get_current_user(token.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    return user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    updates: dict,
    token: str = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Update current user information"""
    user = auth_service.get_current_user(token.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    try:
        return auth_service.update_user(user.id, updates)
    except Exception as e:
        logger.error(f"User update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {str(e)}"
        )

@router.delete("/me")
async def delete_current_user(
    token: str = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Delete current user account"""
    user = auth_service.get_current_user(token.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    try:
        success = auth_service.delete_user(user.id)
        if success:
            return {"message": "User deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    except Exception as e:
        logger.error(f"User deletion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(e)}"
        )

@router.post("/refresh")
async def refresh_token(
    token: str = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh access token"""
    user = auth_service.get_current_user(token.credentials)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    try:
        # Create new access token
        new_token = auth_service._create_access_token(data={"sub": user.id})
        return {"access_token": new_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        ) 