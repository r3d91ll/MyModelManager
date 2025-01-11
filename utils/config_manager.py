from app.model_config import ModelConfig
from app.model_exceptions import ModelFileError, ModelConfigError
import os
import json
import yaml

def load_model_config(model_path: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    config_path = os.path.join(model_path, "model_config.yaml")
    if not os.path.exists(config_path):
        # Try JSON as fallback
        json_path = os.path.join(model_path, "config.json")
        if not os.path.exists(json_path):
            raise ModelConfigError(f"No configuration file found at {config_path} or {json_path}")
        
        try:
            with open(json_path, "r") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ModelConfigError(f"Failed to parse JSON config: {str(e)}")
    else:
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ModelConfigError(f"Failed to parse YAML config: {str(e)}")
    
    try:
        return ModelConfig(**config_data)
    except Exception as e:
        raise ModelConfigError(f"Invalid configuration: {str(e)}")

def apply_model_config(model_path: str, config: ModelConfig):
    """Apply model-specific configuration."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ModelFileError(f"Configuration file not found at {config_path}")

    # Use ModelConfig for validation and default values
    validated_config = config.dict()

    # Update and save the configuration
    with open(config_path, "w") as f:
        json.dump(validated_config, f, indent=2)
