import sqlite3
import secrets
import os
import logging
from typing import Optional
from datetime import datetime
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class AuthManager:
    """Manages API keys and authentication."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'auth.db')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create API keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.commit()
    
    def create_api_key(self, description: str = "") -> str:
        """Create a new API key."""
        try:
            key_id = secrets.token_urlsafe(8)
            api_key = secrets.token_urlsafe(32)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO api_keys (key_id, api_key, description) VALUES (?, ?, ?)",
                    (key_id, api_key, description)
                )
                conn.commit()
            
            logger.info(f"Created new API key with ID: {key_id}")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {str(e)}")
            raise
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key and update last used timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if key exists and is active
                cursor.execute(
                    "SELECT key_id FROM api_keys WHERE api_key = ? AND is_active = TRUE",
                    (api_key,)
                )
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                # Update last used timestamp
                cursor.execute(
                    "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE api_key = ?",
                    (api_key,)
                )
                conn.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return False
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE api_keys SET is_active = FALSE WHERE key_id = ?",
                    (key_id,)
                )
                conn.commit()
                
            logger.info(f"Revoked API key with ID: {key_id}")
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {str(e)}")
            raise

# Global auth manager instance
auth_manager = AuthManager()

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Dependency for validating API key."""
    if api_key_header and auth_manager.validate_api_key(api_key_header):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key"
    )
