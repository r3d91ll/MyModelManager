from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, AutoConfig
import torch
from threading import Thread
import uvicorn
import logging
from dataclasses import dataclass
import pynvml
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model API Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
DOWNLOADS_DIR = os.path.join(os.getcwd(), "downloads")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    stream: Optional[bool] = False

class LoadModelRequest(BaseModel):
    """Request body for loading a model"""
    model_id: str
    config: Optional[Dict[str, Any]] = None

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    context_length: int = 8192
    eval_batch_size: int = 1
    rope_freq_base: float = 1000000.0
    rope_scaling: float = 1.0
    seed: int = 42
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

class GPUInfo:
    def __init__(self):
        """Initialize GPU monitoring."""
        self.device_count = torch.cuda.device_count()
        pynvml.nvmlInit()
        self.handles = []
        self._initialize_gpu_handles()
        self.check_nvlink()
        
    def _initialize_gpu_handles(self):
        """Initialize NVML handles for all available GPUs."""
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self.handles.append(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            logger.info(f"Initialized GPU {i}: {name.decode()}")
    
    def check_nvlink(self):
        """Check NVLink status between GPUs."""
        self.nvlink_enabled = False
        if self.device_count < 2:
            return
            
        try:
            # Check NVLink connectivity between first two GPUs
            handle0 = self.handles[0]
            handle1 = self.handles[1]
            
            # Get NVLink capabilities
            nvlink_caps = pynvml.nvmlDeviceGetNvLinkCapability(handle0, 0, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED)
            if nvlink_caps:
                # Check if NVLink is active
                nvlink_state = pynvml.nvmlDeviceGetNvLinkState(handle0, 0)
                if nvlink_state == pynvml.NVML_NVLINK_STATUS_ACTIVE:
                    self.nvlink_enabled = True
                    logger.info("NVLink is enabled and active between GPUs")
                else:
                    logger.warning("NVLink is supported but not active")
            else:
                logger.warning("NVLink is not supported between GPUs")
                
        except Exception as e:
            logger.warning(f"Error checking NVLink status: {e}")
    
    def get_memory_info(self) -> List[Dict[str, int]]:
        """Get memory information for all GPUs."""
        memory_info = []
        for handle in self.handles:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_info.append({
                'total': info.total,
                'used': info.used,
                'free': info.free
            })
        return memory_info
    
    def calculate_device_map(self, required_memory: int) -> Optional[Dict[str, Any]]:
        """Calculate optimal device mapping based on required memory and available GPUs.
        
        Args:
            required_memory: Estimated memory needed in bytes
            
        Returns:
            Dict with device_map and max_memory settings, or None if insufficient memory
        """
        memory_info = self.get_memory_info()
        total_free_memory = sum(mem['free'] for mem in memory_info)
        
        # Convert to GB for easier comparison
        required_gb = required_memory / (1024**3)
        logger.info(f"Model requires approximately {required_gb:.2f}GB of VRAM")
        
        # If model needs more than 45GB, force tensor parallelism across GPUs
        if required_gb > 45 and self.nvlink_enabled and self.device_count >= 2:
            # Split model across both GPUs
            memory_per_gpu = required_memory / 2
            if all(mem['free'] >= memory_per_gpu for mem in memory_info[:2]):
                logger.info("Using tensor parallelism across both GPUs")
                return {
                    'device_map': 'balanced',
                    'max_memory': {
                        0: f"{memory_info[0]['free'] // (1024**3)}GiB",
                        1: f"{memory_info[1]['free'] // (1024**3)}GiB"
                    }
                }
        
        # Try single GPU first if model is small enough
        for i, mem in enumerate(memory_info):
            if mem['free'] >= required_memory:
                logger.info(f"Allocating model to single GPU {i}")
                return {
                    'device_map': i,
                    'max_memory': {i: f"{mem['free'] // (1024**3)}GiB"}
                }
        
        # If no single GPU has enough memory but combined they do, use tensor parallelism
        if total_free_memory >= required_memory and self.nvlink_enabled and self.device_count >= 2:
            logger.info("Using tensor parallelism across available GPUs")
            return {
                'device_map': 'balanced',
                'max_memory': {
                    i: f"{mem['free'] // (1024**3)}GiB"
                    for i, mem in enumerate(memory_info)
                }
            }
        
        return None

    def __del__(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# Initialize GPU info
gpu_info = GPUInfo()

# Global state for loaded models
class LoadedModel:
    def __init__(self, model_id: str, model, tokenizer, device: int, config: ModelConfig):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.last_used = time.time()

# Track loaded models instead of single current_model
loaded_models: Dict[str, LoadedModel] = {}
gpu_allocations: Dict[int, str] = {}  # GPU device -> model_id

def get_loaded_model(model_id: str) -> Optional[LoadedModel]:
    """Get a loaded model by ID."""
    if model_id in loaded_models:
        model = loaded_models[model_id]
        model.last_used = time.time()
        return model
    return None

def unload_model(model_id: str):
    """Unload a specific model and free its GPU memory."""
    if model_id in loaded_models:
        model = loaded_models[model_id]
        if model.device in gpu_allocations:
            del gpu_allocations[model.device]
        
        del model.model
        del model.tokenizer
        del loaded_models[model_id]
        torch.cuda.empty_cache()
        logger.info(f"Unloaded model {model_id} from GPU {model.device}")

def load_model(model_id: str, config: Optional[ModelConfig] = None) -> LoadedModel:
    """Load a model and its tokenizer."""
    if config is None:
        config = ModelConfig()
    
    # Check if model is already loaded
    existing_model = get_loaded_model(model_id)
    if existing_model is not None:
        return existing_model
        
    models = find_models()
    model_info = next((m for m in models if m["id"] == model_id), None)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    try:
        model_path = model_info['root']
        logger.info(f"Loading model {model_id} from {model_path}")
        
        # Verify model path and files
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
            
        model_files = sorted([f for f in os.listdir(model_path) if f.startswith('model-') and f.endswith('.safetensors')])
        if not model_files:
            raise ValueError(f"No model files found in {model_path}")
        logger.info(f"Found model shards: {model_files}")
        
        # Estimate memory requirements
        total_model_size = sum(os.path.getsize(os.path.join(model_path, f)) for f in model_files)
        estimated_gpu_memory = int(total_model_size * 1.2)  # Add 20% overhead
        
        # Calculate optimal device mapping
        device_config = gpu_info.calculate_device_map(estimated_gpu_memory)
        
        if device_config is None:
            logger.warning("Insufficient GPU memory available, attempting to free memory...")
            # Try to free memory by unloading least recently used models
            while loaded_models and device_config is None:
                lru_model = min(loaded_models.values(), key=lambda m: m.last_used)
                logger.info(f"Unloading least recently used model {lru_model.model_id}")
                unload_model(lru_model.model_id)
                device_config = gpu_info.calculate_device_map(estimated_gpu_memory)
        
        if device_config is None:
            raise ValueError("Insufficient GPU memory available even after unloading models")
        
        # Set random seed
        torch.manual_seed(config.seed)
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer not found at {tokenizer_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load model configuration
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config not found at {config_path}")
            
        model_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Update model configuration
        model_config.max_sequence_length = config.context_length
        model_config.rope_theta = config.rope_freq_base
        model_config.rope_scaling = config.rope_scaling
        
        # Load the model with calculated device mapping
        logger.info(f"Loading model with device configuration: {device_config}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            device_map=device_config['device_map'],
            max_memory=device_config['max_memory'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )
        
        # Set generation parameters
        model.config.temperature = config.temperature
        model.config.top_p = config.top_p
        model.config.top_k = config.top_k
        
        # Create LoadedModel instance
        loaded_model = LoadedModel(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            device=-1 if isinstance(device_config['device_map'], str) else device_config['device_map'],
            config=config
        )
        
        # Store in loaded models
        loaded_models[model_id] = loaded_model
        logger.info(f"Successfully loaded model {model_id}")
        
        return loaded_model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        if model_id in loaded_models:
            unload_model(model_id)
        raise

def find_models() -> List[Dict[str, Any]]:
    """Find all downloaded models."""
    models = []
    downloads_dir = os.path.join(os.getcwd(), "downloads")
    
    if not os.path.exists(downloads_dir):
        return models
        
    # Walk through the downloads directory
    for model_dir in os.listdir(downloads_dir):
        model_path = os.path.join(downloads_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        # Check for variants
        for variant in os.listdir(model_path):
            variant_path = os.path.join(model_path, variant)
            if not os.path.isdir(variant_path):
                continue
                
            # Check for model files
            if any(f.endswith('.safetensors') for f in os.listdir(variant_path)):
                model_format = 'safetensors'
            else:
                continue  # Skip if no recognized model files
                
            models.append({
                'id': f"{model_dir}/{variant}",
                'name': model_dir,
                'variant': variant,
                'format': model_format,
                'root': os.path.abspath(variant_path)  # Use absolute path
            })
                
    return models

@app.get("/v1/models")
async def list_models():
    """List all available models."""
    models = find_models()
    return {
        "data": models,
        "object": "list"
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Generate a chat completion."""
    try:
        # Load model if needed
        model = get_loaded_model(request.model)
        if model is None:
            model = load_model(request.model)
        
        # Prepare conversation
        conversation = ""
        for msg in request.messages:
            if msg.role == "system":
                conversation += f"System: {msg.content}\n"
            elif msg.role == "user":
                conversation += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n"
        conversation += "Assistant: "
        
        # Generate response
        inputs = model.tokenizer(conversation, return_tensors="pt").to(model.model.device)
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            pad_token_id=model.tokenizer.eos_token_id
        )
        response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        response = response.split("Assistant: ")[-1].strip()
        
        return {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": int(os.path.getmtime(model.model.config._name_or_path)),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(outputs[0]) - len(inputs.input_ids[0]),
                "total_tokens": len(outputs[0])
            }
        }
    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/load_model")
async def api_load_model(request: LoadModelRequest):
    """Load a model with optional configuration"""
    try:
        logger.info(f"Received request to load model: {request.model_id}")
        logger.info(f"Config: {request.config}")
        
        # Find model info first
        models = find_models()
        model_info = next((m for m in models if m["id"] == request.model_id), None)
        if not model_info:
            logger.error(f"Model {request.model_id} not found in available models: {[m['id'] for m in models]}")
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            
        logger.info(f"Found model info: {model_info}")
        
        # Convert dict config to ModelConfig
        model_config = ModelConfig()  # Start with defaults
        if request.config:
            try:
                # Update only provided values
                for key, value in request.config.items():
                    if hasattr(model_config, key):
                        setattr(model_config, key, value)
                logger.info(f"Using config: {model_config}")
            except Exception as e:
                logger.error(f"Error setting config values: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid config: {str(e)}")
        
        # Load the model
        load_model(request.model_id, model_config)
        return {"status": "success", "message": f"Model {request.model_id} loaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post("/v1/unload_model")
async def unload_model(model_id: str):
    """Unload a specific model."""
    try:
        unload_model(model_id)
        return {"status": "success", "message": f"Model {model_id} unloaded successfully"}
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/loaded_models")
async def get_loaded_models():
    """Get information about currently loaded models."""
    return {
        model_id: {
            "device": model.device,
            "last_used": model.last_used,
            "config": model.config.__dict__
        }
        for model_id, model in loaded_models.items()
    }

def find_free_port(start_port: int = 8000) -> int:
    """Find a free port starting from start_port."""
    import socket
    port = start_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1

if __name__ == "__main__":
    port = find_free_port()
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
