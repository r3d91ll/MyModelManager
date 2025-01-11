from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ModelConfig(BaseModel):
    """Model configuration settings."""
    context_length: int = Field(4096, description="Maximum context length for the model")
    gpu_device: Optional[int] = Field(None, description="Preferred GPU device to load model on")
    max_batch_size: int = Field(32, description="Maximum batch size for inference")
    quantization: Optional[str] = Field(None, description="Quantization method (e.g. 'q4_k_m', 'q8_0')")
    model_type: str = Field("gguf", description="Model format type (gguf or safetensors)")
    tensor_parallel_size: int = Field(1, description="Number of GPUs to use for tensor parallelism")
    trust_remote_code: bool = Field(False, description="Whether to trust remote code when loading model")
    use_flash_attention: bool = Field(True, description="Whether to use flash attention when available")
    max_new_tokens: int = Field(2048, description="Maximum number of new tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.95, description="Top-p sampling parameter")
    top_k: int = Field(40, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, description="Repetition penalty")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional model-specific parameters")
