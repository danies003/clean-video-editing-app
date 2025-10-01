"""
Redis Validation Service
Prevents and detects Redis corruption issues before they cause system failures.
"""

import redis
import logging
from typing import Dict, List, Any, Optional
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class RedisValidator:
    """Validates Redis data integrity and prevents corruption."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.settings = get_settings()
        
    def validate_redis_health(self) -> Dict[str, Any]:
        """Comprehensive Redis health check."""
        health_status = {
            "overall": "healthy",
            "connection": False,
            "key_types": {},
            "corruption_detected": False,
            "issues": []
        }
        
        try:
            # Test connection
            self.redis.ping()
            health_status["connection"] = True
            
            # Validate critical Redis keys
            self._validate_critical_keys(health_status)
            
            # Check for data type mismatches
            self._check_data_types(health_status)
            
            if health_status["issues"]:
                health_status["overall"] = "degraded"
                health_status["corruption_detected"] = True
                
        except Exception as e:
            health_status["overall"] = "unhealthy"
            health_status["issues"].append(f"Connection failed: {str(e)}")
            logger.error(f"Redis health check failed: {e}")
            
        return health_status
    
    def _validate_critical_keys(self, health_status: Dict[str, Any]) -> None:
        """Validate critical Redis keys that are essential for system operation."""
        critical_keys = {
            "rq:queues": "set",
            "rq:workers": "set",
            "rq:clean_registries": "string"
        }
        
        for key, expected_type in critical_keys.items():
            try:
                if self.redis.exists(key):
                    actual_type = self.redis.type(key)
                    # Handle both string and bytes return types
                    if isinstance(actual_type, bytes):
                        actual_type = actual_type.decode('utf-8')
                    health_status["key_types"][key] = actual_type
                    
                    if actual_type != expected_type:
                        issue = f"Key {key} has wrong type: expected {expected_type}, got {actual_type}"
                        health_status["issues"].append(issue)
                        logger.warning(f"⚠️ {issue}")
                        
                        # Attempt to fix the corrupted key
                        self._fix_corrupted_key(key, expected_type)
                else:
                    health_status["key_types"][key] = "missing"
                    
            except Exception as e:
                issue = f"Failed to validate key {key}: {str(e)}"
                health_status["issues"].append(issue)
                logger.error(f"❌ {issue}")
    
    def _check_data_types(self, health_status: Dict[str, Any]) -> None:
        """Check for common data type corruption patterns."""
        try:
            # Check failed jobs queue
            failed_key = "rq:failed:video_editing"
            if self.redis.exists(failed_key):
                key_type = self.redis.type(failed_key)
                # Handle both string and bytes return types
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                health_status["key_types"][failed_key] = key_type
                
                if key_type != "list":
                    issue = f"Failed jobs queue has wrong type: expected list, got {key_type}"
                    health_status["issues"].append(issue)
                    logger.warning(f"⚠️ {issue}")
                    
                    # Fix the corrupted failed jobs queue
                    self._fix_corrupted_key(failed_key, "list")
                    
        except Exception as e:
            issue = f"Failed to check data types: {str(e)}"
            health_status["issues"].append(issue)
            logger.error(f"❌ {issue}")
    
    def _fix_corrupted_key(self, key: str, expected_type: str) -> bool:
        """Attempt to fix a corrupted Redis key."""
        try:
            logger.info(f"🔧 Attempting to fix corrupted key: {key} -> {expected_type}")
            
            if expected_type == "list":
                # For lists, we need to preserve existing data if possible
                key_type = self.redis.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                if key_type == "zset":
                    # Convert zset to list
                    items = self.redis.zrange(key, 0, -1)
                    self.redis.delete(key)
                    if items:
                        self.redis.lpush(key, *items)
                    logger.info(f"✅ Converted zset to list for key: {key}")
                    
            elif expected_type == "set":
                # For sets, ensure it's actually a set
                key_type = self.redis.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                if key_type != "set":
                    # Recreate as empty set
                    self.redis.delete(key)
                    self.redis.sadd(key, "placeholder")
                    self.redis.srem(key, "placeholder")
                    logger.info(f"✅ Recreated set for key: {key}")
                    
            elif expected_type == "string":
                # For strings, ensure it's actually a string
                key_type = self.redis.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')
                if key_type != "string":
                    self.redis.delete(key)
                    self.redis.set(key, "")
                    logger.info(f"✅ Recreated string for key: {key}")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix corrupted key {key}: {e}")
            return False
    
    def prevent_corruption(self) -> bool:
        """Proactive corruption prevention."""
        try:
            # Validate settings before they can cause corruption
            if not hasattr(self.settings, 'allowed_origins'):
                logger.error("❌ Settings missing allowed_origins attribute")
                return False
                
            if not isinstance(self.settings.allowed_origins, list):
                logger.error(f"❌ Settings allowed_origins is not a list: {type(self.settings.allowed_origins)}")
                return False
                
            # Validate Redis connection
            self.redis.ping()
            
            # Ensure critical keys exist with correct types
            self._ensure_critical_keys()
            
            logger.info("✅ Corruption prevention checks passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Corruption prevention failed: {e}")
            return False
    
    def _ensure_critical_keys(self) -> None:
        """Ensure critical Redis keys exist with correct types."""
        critical_keys = {
            "rq:queues": "set",
            "rq:workers": "set",
            "rq:clean_registries": "string"
        }
        
        for key, expected_type in critical_keys.items():
            try:
                if not self.redis.exists(key):
                    if expected_type == "set":
                        self.redis.sadd(key, "placeholder")
                        self.redis.srem(key, "placeholder")
                    elif expected_type == "string":
                        self.redis.set(key, "")
                    logger.info(f"✅ Created missing critical key: {key}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create critical key {key}: {e}")
    
    def cleanup_corrupted_keys(self) -> Dict[str, int]:
        """Clean up corrupted Redis keys and return count of cleaned keys."""
        cleaned_count = 0
        cleaned_keys = []
        
        try:
            # Get all keys
            all_keys = self.redis.keys("*")
            
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                try:
                    # Check if key is corrupted
                    if self._is_key_corrupted(key_str):
                        logger.warning(f"🗑️ Cleaning up corrupted key: {key_str}")
                        self.redis.delete(key_str)
                        cleaned_count += 1
                        cleaned_keys.append(key_str)
                        
                except Exception as e:
                    logger.error(f"❌ Error checking key {key_str}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            
        logger.info(f"🧹 Cleaned up {cleaned_count} corrupted keys: {cleaned_keys}")
        return {"cleaned_count": cleaned_count, "cleaned_keys": cleaned_keys}
    
    def _is_key_corrupted(self, key: str) -> bool:
        """Check if a Redis key is corrupted."""
        try:
            key_type = self.redis.type(key)
            # Handle both string and bytes return types
            if isinstance(key_type, bytes):
                key_type = key_type.decode('utf-8')
            
            # Define expected types for known keys
            expected_types = {
                "rq:failed:video_editing": "list",
                "rq:queues": "set",
                "rq:workers": "set",
                "rq:clean_registries": "string"
            }
            
            if key in expected_types:
                return key_type != expected_types[key]
                
            return False
            
        except Exception:
            return True  # Consider corrupted if we can't check
