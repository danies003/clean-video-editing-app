from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from enum import Enum

class AuthProvider(str, Enum):
    EMAIL = "email"
    GOOGLE = "google"
    FACEBOOK = "facebook"

class UserSignUp(BaseModel):
    name: str
    email: EmailStr
    password: str
    provider: AuthProvider = AuthProvider.EMAIL

class UserSignIn(BaseModel):
    email: EmailStr
    password: str

class SocialAuth(BaseModel):
    provider: AuthProvider
    token: Optional[str] = None  # For ID token flow
    code: Optional[str] = None   # For authorization code flow
    redirect_uri: Optional[str] = None  # For authorization code flow
    email: Optional[EmailStr] = None
    name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    name: str
    email: EmailStr
    provider: AuthProvider
    created_at: datetime
    updated_at: datetime

class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None 