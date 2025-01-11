from pydantic import BaseModel, Field
from typing import Optional, List, Union

class ModelConfig(BaseModel):
    """Configuration for loading and running a model."""
    # Model architecture settings
    model_type: str = Field("llama", description="Type of model (e.g., llama, gpt2)")
    architecture: str = Field("llama2", description="Model architecture")
    
    # Context and batch settings
    context_length: int = Field(4096, description="Maximum context length")
    max_batch_size: int = Field(512, description="Maximum batch size for processing")
    
    # Generation parameters
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(40, ge=0, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, ge=0.0, description="Repetition penalty")
    max_new_tokens: int = Field(2048, ge=1, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    
    # Hardware settings
    gpu_device: Optional[Union[int, List[int]]] = Field(None, description="GPU device(s) to use")
    use_flash_attention: bool = Field(True, description="Whether to use flash attention")
    trust_remote_code: bool = Field(True, description="Whether to trust remote code")
