import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from app.models.auth import UserSignUp, UserSignIn, SocialAuth, UserResponse, AuthResponse, AuthProvider
from app.services.oauth_service import oauth_service
import redis
import json
import os

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    def __init__(self):
        # Use REDIS_URL if available (Railway), otherwise fall back to individual host/port
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=0,
                decode_responses=True
            )
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def _create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def _get_user_key(self, user_id: str) -> str:
        """Get Redis key for user data"""
        return f"user:{user_id}"
    
    def _get_user_by_email_key(self, email: str) -> str:
        """Get Redis key for user email lookup"""
        return f"user_email:{email.lower()}"
    
    def _get_user_by_provider_id_key(self, provider: str, provider_id: str) -> str:
        """Get Redis key for provider ID lookup"""
        return f"user_provider:{provider}:{provider_id}"
    
    def register_user(self, user_data: UserSignUp) -> AuthResponse:
        """Register a new user"""
        # Check if user already exists
        email_key = self._get_user_by_email_key(user_data.email)
        if self.redis_client.exists(email_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Hash password if using email provider
        hashed_password = None
        if user_data.provider == AuthProvider.EMAIL:
            hashed_password = self._hash_password(user_data.password)
        
        user_data_dict = {
            "id": user_id,
            "name": user_data.name,
            "email": user_data.email.lower(),
            "password_hash": hashed_password,
            "provider": user_data.provider.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }
        
        # Store user data in Redis
        user_key = self._get_user_key(user_id)
        self.redis_client.setex(user_key, 86400 * 30, json.dumps(user_data_dict))  # 30 days TTL
        self.redis_client.setex(email_key, 86400 * 30, user_id)  # 30 days TTL
        
        # Create access token
        access_token = self._create_access_token(data={"sub": user_id})
        
        # Create response
        user_response = UserResponse(
            id=user_id,
            name=user_data.name,
            email=user_data.email,
            provider=user_data.provider,
            created_at=now,
            updated_at=now
        )
        
        return AuthResponse(user=user_response, access_token=access_token)
    
    def authenticate_user(self, user_data: UserSignIn) -> AuthResponse:
        """Authenticate a user with email and password"""
        # Find user by email
        email_key = self._get_user_by_email_key(user_data.email)
        user_id = self.redis_client.get(email_key)
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Get user data
        user_key = self._get_user_key(user_id)
        user_data_str = self.redis_client.get(user_key)
        
        if not user_data_str:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        user_dict = json.loads(user_data_str)
        
        # Verify password
        if not self._verify_password(user_data.password, user_dict["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create access token
        access_token = self._create_access_token(data={"sub": user_id})
        
        # Create response
        user_response = UserResponse(
            id=user_dict["id"],
            name=user_dict["name"],
            email=user_dict["email"],
            provider=AuthProvider(user_dict["provider"]),
            created_at=datetime.fromisoformat(user_dict["created_at"]),
            updated_at=datetime.fromisoformat(user_dict["updated_at"])
        )
        
        return AuthResponse(user=user_response, access_token=access_token)
    
    def authenticate_social_user(self, social_data: SocialAuth) -> AuthResponse:
        """Authenticate a user with social provider using real OAuth verification"""
        try:
            # Verify token/code with OAuth provider
            if social_data.provider == AuthProvider.GOOGLE:
                if social_data.code and social_data.redirect_uri:
                    # Handle authorization code flow
                    user_info = oauth_service.verify_google_code(social_data.code, social_data.redirect_uri)
                elif social_data.token:
                    # Handle ID token flow
                    user_info = oauth_service.verify_google_token(social_data.token)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Either code or token must be provided for Google OAuth"
                    )
            elif social_data.provider == AuthProvider.FACEBOOK:
                # Support both code flow (redirect) and manual token flow
                if social_data.code and social_data.redirect_uri:
                    user_info = oauth_service.verify_facebook_code(social_data.code, social_data.redirect_uri)
                elif social_data.token:
                    user_info = oauth_service.verify_facebook_token(social_data.token)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Either code or token must be provided for Facebook OAuth"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported social provider"
                )
            
            # Check if user already exists by provider ID
            provider_key = self._get_user_by_provider_id_key(
                social_data.provider.value, 
                user_info["id"]
            )
            existing_user_id = self.redis_client.get(provider_key)
            
            if existing_user_id:
                # User exists, get their data
                user_key = self._get_user_key(existing_user_id)
                user_data_str = self.redis_client.get(user_key)
                user_dict = json.loads(user_data_str)
            else:
                # Create new user
                user_id = str(uuid.uuid4())
                now = datetime.utcnow()
                
                user_dict = {
                    "id": user_id,
                    "name": user_info.get("name", f"{social_data.provider.value.title()} User"),
                    "email": user_info.get("email", ""),
                    "password_hash": None,
                    "provider": social_data.provider.value,
                    "provider_id": user_info["id"],
                    "picture": user_info.get("picture", ""),
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat()
                }
                
                # Store user data in Redis
                user_key = self._get_user_key(user_id)
                email_key = self._get_user_by_email_key(user_dict["email"]) if user_dict["email"] else None
                
                self.redis_client.setex(user_key, 86400 * 30, json.dumps(user_dict))
                self.redis_client.setex(provider_key, 86400 * 30, user_id)
                
                if email_key:
                    self.redis_client.setex(email_key, 86400 * 30, user_id)
            
            # Create access token
            access_token = self._create_access_token(data={"sub": user_dict["id"]})
            
            # Create response
            user_response = UserResponse(
                id=user_dict["id"],
                name=user_dict["name"],
                email=user_dict["email"],
                provider=AuthProvider(user_dict["provider"]),
                created_at=datetime.fromisoformat(user_dict["created_at"]),
                updated_at=datetime.fromisoformat(user_dict["updated_at"])
            )
            
            return AuthResponse(user=user_response, access_token=access_token)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Social authentication failed: {str(e)}"
            )
    
    def get_current_user(self, token: str) -> Optional[UserResponse]:
        """Get current user from JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
        except jwt.PyJWTError:
            return None
        
        # Get user data from Redis
        user_key = self._get_user_key(user_id)
        user_data_str = self.redis_client.get(user_key)
        
        if not user_data_str:
            return None
        
        user_dict = json.loads(user_data_str)
        
        return UserResponse(
            id=user_dict["id"],
            name=user_dict["name"],
            email=user_dict["email"],
            provider=AuthProvider(user_dict["provider"]),
            created_at=datetime.fromisoformat(user_dict["created_at"]),
            updated_at=datetime.fromisoformat(user_dict["updated_at"])
        )
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> UserResponse:
        """Update user information"""
        user_key = self._get_user_key(user_id)
        user_data_str = self.redis_client.get(user_key)
        
        if not user_data_str:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user_dict = json.loads(user_data_str)
        user_dict.update(updates)
        user_dict["updated_at"] = datetime.utcnow().isoformat()
        
        # Update in Redis
        self.redis_client.setex(user_key, 86400 * 30, json.dumps(user_dict))
        
        return UserResponse(
            id=user_dict["id"],
            name=user_dict["name"],
            email=user_dict["email"],
            provider=AuthProvider(user_dict["provider"]),
            created_at=datetime.fromisoformat(user_dict["created_at"]),
            updated_at=datetime.fromisoformat(user_dict["updated_at"])
        )
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user"""
        user_key = self._get_user_key(user_id)
        user_data_str = self.redis_client.get(user_key)
        
        if not user_data_str:
            return False
        
        user_dict = json.loads(user_data_str)
        email_key = self._get_user_by_email_key(user_dict["email"])
        
        # Delete from Redis
        self.redis_client.delete(user_key)
        self.redis_client.delete(email_key)
        
        return True 