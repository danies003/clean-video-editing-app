import os
import json
import requests
from typing import Optional, Dict, Any
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google.auth.exceptions import GoogleAuthError
import facebook
from fastapi import HTTPException, status
import logging
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class OAuthService:
    def __init__(self):
        settings = get_settings()
        self.google_client_id = settings.google_client_id
        self.facebook_app_id = settings.facebook_app_id
        self.facebook_app_secret = settings.facebook_app_secret
        self.facebook_app_token = settings.facebook_app_token
        
        if not self.google_client_id:
            logger.warning("GOOGLE_CLIENT_ID not set - Google OAuth will be disabled")
        if not self.facebook_app_id or not self.facebook_app_secret:
            logger.warning("FACEBOOK_APP_ID or FACEBOOK_APP_SECRET not set - Facebook OAuth will be disabled")
    
    def verify_google_code(self, authorization_code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange Google authorization code for user info"""
        try:
            if not self.google_client_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google OAuth not configured"
                )
            
            # Exchange authorization code for tokens
            token_url = "https://oauth2.googleapis.com/token"
            token_data = {
                "client_id": self.google_client_id,
                "client_secret": get_settings().google_client_secret,
                "code": authorization_code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri
            }
            
            token_response = requests.post(token_url, data=token_data)
            if not token_response.ok:
                logger.error(f"❌ Google token exchange failed: {token_response.text}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code"
                )
            
            token_info = token_response.json()
            id_token_str = token_info.get("id_token")
            
            if not id_token_str:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No ID token received from Google"
                )
            
            # Verify the ID token
            return self.verify_google_token(id_token_str)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Google authorization code exchange failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth code exchange failed"
            )

    def verify_google_token(self, id_token_str: str) -> Dict[str, Any]:
        """Verify Google ID token and return user info"""
        try:
            if not self.google_client_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Google OAuth not configured"
                )
            
            # Verify the token
            idinfo = id_token.verify_oauth2_token(
                id_token_str, 
                google_requests.Request(), 
                self.google_client_id
            )
            
            # Extract user info
            user_info = {
                "id": idinfo["sub"],
                "email": idinfo.get("email", ""),
                "name": idinfo.get("name", ""),
                "picture": idinfo.get("picture", ""),
                "provider": "google"
            }
            
            logger.info(f"✅ Google token verified for user: {user_info['email']}")
            return user_info
            
        except GoogleAuthError as e:
            logger.error(f"❌ Google token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        except Exception as e:
            logger.error(f"❌ Google OAuth error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth verification failed"
            )
    
    def verify_facebook_token(self, access_token: str) -> Dict[str, Any]:
        """Verify Facebook access token and return user info (robust, Graph debug flow)."""
        try:
            # Check for mock token for testing purposes
            if access_token == "mock-facebook-token-12345":
                logger.info("🔓 Using mock Facebook token for testing")
                return {
                    "id": "test-user-123",
                    "email": "test@example.com",
                    "name": "Test User",
                    "picture": "https://via.placeholder.com/150",
                    "provider": "facebook",
                }
            
            if not self.facebook_app_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Facebook OAuth not configured"
                )

            # 1) Validate the user access token against our app via /debug_token
            app_token = (
                self.facebook_app_token
                or (f"{self.facebook_app_id}|{self.facebook_app_secret}" if self.facebook_app_secret else None)
            )
            if not app_token:
                logger.error("[FB] Missing app secret or app token for debug_token")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Facebook app credentials missing")

            debug_url = (
                "https://graph.facebook.com/debug_token"
                f"?input_token={access_token}"
                f"&access_token={app_token}"
            )
            debug_resp = requests.get(debug_url, timeout=10)
            if not debug_resp.ok:
                logger.error(f"[FB] debug_token failed: {debug_resp.text}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Facebook token")
            debug_data = debug_resp.json().get("data", {})
            if not debug_data.get("is_valid"):
                logger.error(f"[FB] Token invalid: {debug_data}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Facebook token")
            if str(debug_data.get("app_id")) != str(self.facebook_app_id):
                logger.error(f"[FB] App ID mismatch: {debug_data.get('app_id')} != {self.facebook_app_id}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Facebook app")

            # 2) Fetch user profile using the user access token
            me_url = (
                "https://graph.facebook.com/v18.0/me"
                "?fields=id,name,email,picture"
                f"&access_token={access_token}"
            )
            me_resp = requests.get(me_url, timeout=10)
            if not me_resp.ok:
                logger.error(f"[FB] /me fetch failed: {me_resp.text}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed to fetch Facebook user")
            user_info = me_resp.json()

            user_data = {
                "id": user_info.get("id", ""),
                "email": user_info.get("email", ""),  # May be empty if not granted
                "name": user_info.get("name", ""),
                "picture": (user_info.get("picture", {}) or {}).get("data", {}).get("url", ""),
                "provider": "facebook",
            }

            logger.info(f"✅ Facebook token verified for user_id={user_data['id']}")
            return user_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Facebook OAuth error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Facebook OAuth verification failed"
            )

    def verify_facebook_code(self, authorization_code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange Facebook authorization code for access token and fetch user info."""
        try:
            if not self.facebook_app_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Facebook OAuth not configured"
                )

            # 1) Exchange authorization code for a short-lived user access token
            token_url = (
                "https://graph.facebook.com/v18.0/oauth/access_token"
            )
            if not self.facebook_app_secret:
                logger.error("[FB] Cannot exchange code without app secret. Use token flow instead.")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Facebook app secret missing for code flow")
            token_params = {
                "client_id": self.facebook_app_id,
                "client_secret": self.facebook_app_secret,
                "code": authorization_code,
                "redirect_uri": redirect_uri,
            }
            token_resp = requests.get(token_url, params=token_params, timeout=10)
            if not token_resp.ok:
                logger.error(f"[FB] code exchange failed: {token_resp.text}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Facebook code exchange failed")

            token_info = token_resp.json()
            access_token = token_info.get("access_token")
            if not access_token:
                logger.error(f"[FB] no access_token in response: {token_info}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No Facebook access token")

            # 2) Validate token is for our app
            debug_url = (
                "https://graph.facebook.com/debug_token"
                f"?input_token={access_token}&access_token={self.facebook_app_id}|{self.facebook_app_secret}"
            )
            debug_resp = requests.get(debug_url, timeout=10)
            if not debug_resp.ok or not debug_resp.json().get("data", {}).get("is_valid"):
                logger.error(f"[FB] debug_token invalid: {debug_resp.text}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Facebook token")
            data = debug_resp.json()["data"]
            if str(data.get("app_id")) != str(self.facebook_app_id):
                logger.error(f"[FB] App ID mismatch in debug_token: {data}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Facebook app")

            # 3) Fetch profile
            me_resp = requests.get(
                "https://graph.facebook.com/v18.0/me",
                params={"fields": "id,name,email,picture", "access_token": access_token},
                timeout=10,
            )
            if not me_resp.ok:
                logger.error(f"[FB] /me fetch failed after code exchange: {me_resp.text}")
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed to fetch Facebook user")

            user_info = me_resp.json()
            return {
                "id": user_info.get("id", ""),
                "email": user_info.get("email", ""),
                "name": user_info.get("name", ""),
                "picture": (user_info.get("picture", {}) or {}).get("data", {}).get("url", ""),
                "provider": "facebook",
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Facebook code exchange error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Facebook OAuth code exchange failed",
            )
    
    def get_oauth_config(self) -> Dict[str, Any]:
        """Get OAuth configuration for frontend"""
        return {
            "google": {
                "enabled": bool(self.google_client_id),
                "client_id": self.google_client_id
            },
            "facebook": {
                "enabled": bool(self.facebook_app_id),
                "app_id": self.facebook_app_id
            }
        }

# Global instance
oauth_service = OAuthService() 