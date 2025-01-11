"""Custom exceptions for the model manager application."""

class ModelManagerError(Exception):
    """Base exception class for model manager errors."""
    pass

class ModelNotFoundError(ModelManagerError):
    """Raised when a requested model is not found."""
    pass

class ModelFileError(ModelManagerError):
    """Raised when there is an error with model files."""
    pass

class ModelLoadError(ModelManagerError):
    """Raised when there is an error loading a model."""
    pass

class GPUMemoryError(ModelManagerError):
    """Raised when there is insufficient GPU memory."""
    pass

class ModelNotLoadedError(ModelManagerError):
    """Raised when trying to use a model that is not loaded."""
    pass

class ContextLengthError(ModelManagerError):
    """Raised when context length is invalid or exceeds model limits."""
    pass

class ModelConfigError(Exception):
    """Raised when there is an error in model configuration."""
    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 400  # Bad Request

class ConfigurationError(ModelManagerError):
    """Raised when there is an error in model configuration."""
    pass
