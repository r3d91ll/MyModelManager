import sqlite3
import os
import json
import yaml
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized registry for model information and configurations."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'model_registry.db')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    config_path TEXT,
                    default_config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def register_model(self, 
                      model_id: str, 
                      model_path: str, 
                      model_type: str,
                      config_path: Optional[str] = None,
                      default_config: Optional[Dict[str, Any]] = None):
        """Register a new model in the registry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert default config to JSON string if provided
                default_config_str = json.dumps(default_config) if default_config else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO models 
                    (model_id, model_path, model_type, config_path, default_config)
                    VALUES (?, ?, ?, ?, ?)
                """, (model_id, model_path, model_type, config_path, default_config_str))
                
                conn.commit()
                logger.info(f"Registered model {model_id} at {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {str(e)}")
            raise
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT model_id, model_path, model_type, config_path, default_config
                    FROM models WHERE model_id = ?
                """, (model_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                model_info = {
                    "model_id": row[0],
                    "model_path": row[1],
                    "model_type": row[2],
                    "config_path": row[3],
                    "default_config": json.loads(row[4]) if row[4] else None
                }
                
                return model_info
                
        except Exception as e:
            logger.error(f"Failed to get info for model {model_id}: {str(e)}")
            raise
    
    def list_models(self) -> list[Dict[str, Any]]:
        """List all registered models."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT model_id, model_path, model_type FROM models")
                
                models = []
                for row in cursor.fetchall():
                    models.append({
                        "model_id": row[0],
                        "model_path": row[1],
                        "model_type": row[2]
                    })
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            raise
    
    def load_model_config(self, model_id: str) -> Dict[str, Any]:
        """Load configuration for a model, combining default and custom configs."""
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise KeyError(f"Model {model_id} not found in registry")
        
        # Start with default config if available
        config = model_info.get("default_config", {}) or {}
        
        # If config path exists, load and merge with defaults
        if model_info["config_path"] and os.path.exists(model_info["config_path"]):
            try:
                with open(model_info["config_path"]) as f:
                    if model_info["config_path"].endswith('.yaml'):
                        custom_config = yaml.safe_load(f)
                    else:
                        custom_config = json.load(f)
                
                # Update default config with custom values
                config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load custom config for {model_id}: {str(e)}")
        
        return config
    
    def update_model_config(self, model_id: str, config: Dict[str, Any]):
        """Update the default configuration for a model."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE models SET default_config = ?
                    WHERE model_id = ?
                """, (json.dumps(config), model_id))
                
                if cursor.rowcount == 0:
                    raise KeyError(f"Model {model_id} not found in registry")
                
                conn.commit()
                logger.info(f"Updated config for model {model_id}")
                
        except Exception as e:
            logger.error(f"Failed to update config for model {model_id}: {str(e)}")
            raise
