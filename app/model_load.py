from utils.file_manager import find_model_files
from utils.memory_manager import estimate_model_memory, allocate_memory
from utils.config_manager import apply_model_config
from utils.model_registry import ModelRegistry
from app.model_exceptions import ModelNotFoundError, ModelFileError, ModelLoadError, GPUMemoryError
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import asyncio
import logging
import os
from typing import Optional, Dict, List
import torch
import pynvml
import time

logger = logging.getLogger(__name__)

# Global model registry
MODEL_REGISTRY = {}
model_db = ModelRegistry()

class LoadModel:
    """Handles loading models into GPU VRAM."""

    def __init__(self, model_id: str, config):
        self.model_id = model_id
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.last_used = None
        self._loading = False
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        """Ensure the model is loaded before use."""
        global MODEL_REGISTRY
        
        async with self._lock:
            if self.model_id in MODEL_REGISTRY:
                self.model = MODEL_REGISTRY[self.model_id]
                self._loaded = True
                logger.info(f"Using existing model instance for {self.model_id}")
                return

            if not self._loading:
                self._loading = True
                await self._load_model()

    async def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            # Get model info from registry
            model_info = model_db.get_model_info(self.model_id)
            if not model_info:
                raise ModelNotFoundError(f"Model {self.model_id} not found in registry")
            
            model_path = model_info["model_path"]
            model_type = model_info["model_type"]
            
            if model_type == 'gguf':
                # Load GGUF model
                logger.info(f"Loading GGUF model from {model_path}")

                # Import llama_cpp here to avoid loading CUDA unnecessarily
                from llama_cpp import Llama

                # Get GPU configuration
                gpu_device = self.config.gpu_device
                n_gpu_layers = -1  # Default to all layers

                if gpu_device is not None:
                    if not isinstance(gpu_device, list):
                        gpu_device = [gpu_device]
                    
                    # Initialize NVML
                    try:
                        pynvml.nvmlInit()
                        for device_id in gpu_device:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            logger.info(f"GPU {device_id} - Total memory: {info.total / 1024**2:.0f}MB, "
                                      f"Free memory: {info.free / 1024**2:.0f}MB")
                    except Exception as e:
                        logger.warning(f"Failed to get GPU info: {e}")

                    # Convert GPU device list to CUDA device string
                    cuda_devices = ",".join(map(str, gpu_device))
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
                    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
                    logger.info(f"Using GPU devices: {cuda_devices}")

                # Find GGUF file
                gguf_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
                if not gguf_files:
                    raise ModelFileError("No GGUF file found in model directory")
                gguf_path = os.path.join(model_path, gguf_files[0])

                # Load the model into GPU memory
                try:
                    self.model = Llama(
                        model_path=gguf_path,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=self.config.context_length,
                        n_batch=self.config.max_batch_size,
                        temperature=self.config.temperature,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        repeat_penalty=self.config.repetition_penalty,
                        max_tokens=self.config.max_new_tokens,
                        seed=42,
                        verbose=True
                    )
                    self._loaded = True
                    logger.info("Successfully loaded GGUF model")
                except Exception as e:
                    raise ModelLoadError(f"Failed to load GGUF model: {str(e)}")
                    
            elif model_type == 'safetensors':
                # Load safetensors model
                logger.info(f"Loading safetensors model from {model_path}")
                
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model_config = AutoConfig.from_pretrained(model_path)
                    
                    # Apply configuration
                    model_config.max_position_embeddings = self.config.context_length
                    model_config.max_sequence_length = self.config.context_length
                    
                    # Load model with device map
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        config=model_config,
                        device_map="auto",
                        trust_remote_code=self.config.trust_remote_code,
                        use_flash_attention_2=self.config.use_flash_attention
                    )
                    self._loaded = True
                    logger.info("Successfully loaded safetensors model")
                except Exception as e:
                    raise ModelLoadError(f"Failed to load safetensors model: {str(e)}")
            else:
                raise ModelLoadError(f"Unsupported model type: {model_type}")
                
            # Add to registry
            MODEL_REGISTRY[self.model_id] = self.model
            self.last_used = time.time()
            
        except Exception as e:
            self._loading = False
            self._loaded = False
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def unload_model(self) -> None:
        """Unload the model and free resources."""
        if not self._loaded:
            raise ModelLoadError("Model is not loaded")

        try:
            logger.info(f"Unloading model {self.model_id}")
            del self.model
            self._loaded = False
        except Exception as e:
            logger.error(f"Error unloading model {self.model_id}: {e}")
            raise
