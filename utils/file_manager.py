# file_manager.py

import os
from app.model_exceptions import ModelFileError

def find_model_files(model_id: str) -> str:
    """Find model files for a given model ID."""
    model_path = os.path.join("downloads", model_id)
    if not os.path.exists(model_path):
        raise ModelFileError(f"Model path {model_path} does not exist.")
    return model_path
