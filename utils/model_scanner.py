import os
import logging
from typing import Optional
from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

def scan_models(downloads_dir: str, registry: ModelRegistry):
    """Scan the downloads directory and register found models."""
    for model_id in os.listdir(downloads_dir):
        model_dir = os.path.join(downloads_dir, model_id)
        if not os.path.isdir(model_dir):
            continue
            
        # Look for model files
        model_path = None
        model_type = None
        config_path = None
        
        # Check for model config
        yaml_config = os.path.join(model_dir, "model_config.yaml")
        json_config = os.path.join(model_dir, "config.json")
        if os.path.exists(yaml_config):
            config_path = yaml_config
        elif os.path.exists(json_config):
            config_path = json_config
            
        # Check for GGUF files
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.gguf'):
                    model_path = os.path.dirname(os.path.join(root, file))
                    model_type = 'gguf'
                    break
            if model_path:
                break
                
        # Check for safetensors
        if not model_path:
            subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(model_dir, subdir)
                if os.path.exists(os.path.join(subdir_path, "config.json")):
                    model_path = subdir_path
                    model_type = 'safetensors'
                    break
        
        if model_path and model_type:
            # Default configurations based on model type
            default_config = {
                "context_length": 4096,
                "max_batch_size": 512,
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "max_new_tokens": 2048,
                "gpu_device": [0],
                "trust_remote_code": True,
                "use_flash_attention": False
            }
            
            # Register the model
            registry.register_model(
                model_id=model_id,
                model_path=model_path,
                model_type=model_type,
                config_path=config_path,
                default_config=default_config
            )
            logger.info(f"Registered {model_type} model: {model_id}")
        else:
            logger.warning(f"No valid model files found for {model_id}")

def initialize_registry(base_dir: Optional[str] = None) -> ModelRegistry:
    """Initialize the model registry and scan for models."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Scan downloads directory
    downloads_dir = os.path.join(base_dir, "downloads")
    if os.path.exists(downloads_dir):
        scan_models(downloads_dir, registry)
    else:
        logger.warning(f"Downloads directory not found at {downloads_dir}")
    
    return registry
