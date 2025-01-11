from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Tuple
import asyncio
import logging
import os
import time
import sys
import uvicorn
from contextlib import asynccontextmanager

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from app.model_exceptions import (
    ModelNotFoundError, ModelFileError, ModelLoadError, 
    GPUMemoryError, ContextLengthError, ModelConfigError
)
from app.model_config import ModelConfig
from app.model_load import LoadModel
from app.model_chat import ChatCompletionRequest, ChatCompletionResponse, ModelChat
import api.gpu_api as gpu_api
from utils.model_registry import ModelRegistry
from utils.model_scanner import initialize_registry
from utils.auth_manager import get_api_key, auth_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize GPU manager
gpu_manager = gpu_api.GPUManager()

# Global model loader instance
model_loader: Optional[LoadModel] = None
model_loader_lock = asyncio.Lock()

async def get_model_loader(model_id: str, config: Optional[ModelConfig] = None) -> LoadModel:
    """Get or create model loader instance."""
    global model_loader
    
    async with model_loader_lock:
        if model_loader is None or model_loader.model_id != model_id:
            if model_loader is not None:
                await model_loader.unload_model()
            model_loader = LoadModel(model_id, config or ModelConfig())
            await model_loader.ensure_loaded()
        return model_loader

# Initialize FastAPI app
app = FastAPI(title="Model Manager API")
app.include_router(gpu_api.router)

# Initialize model registry
model_db = initialize_registry()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOWNLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloads")
DEFAULT_MAX_CONTEXT = 100000
DEFAULT_BASE_CONTEXT = 2048
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_ROPE_FREQ = 10000
TENSOR_PARALLEL_THRESHOLD = int(os.getenv("TENSOR_PARALLEL_THRESHOLD", 45 * 1024**3))
VRAM_OVERHEAD_FACTOR = 1.2  # 20% overhead for GPU memory estimates

class ModelFileError(ModelLoadError):
    """Raised when model files are missing or invalid"""
    status_code = 404

class ChatMessage(BaseModel):
    """A chat message with role and content."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)

class LoadModelRequest(BaseModel):
    """Request body for loading a model."""
    model_id: str = Field(..., description="The ID of the model to load")
    config: Optional[ModelConfig] = Field(None, description="Optional model configuration")

    @field_validator("config", mode="before")
    def ensure_config(cls, v: Optional[ModelConfig]) -> ModelConfig:
        """Ensure config is always present with at least default values."""
        return v or ModelConfig()

class ModelMemoryInfo(BaseModel):
    """Model memory usage information."""
    allocated_gb: float
    reserved_gb: float
    cached_gb: float
    total_gb: float
    last_used: str

class LoadedModelInfo(BaseModel):
    """Information about a loaded model."""
    model_id: str
    memory: ModelMemoryInfo
    device_map: Union[str, Dict[str, int]]
    context_length: int
    last_used: str

class UnloadModelRequest(BaseModel):
    """Request to unload a model."""
    model_id: str
    force: bool = Field(False, description="Force unload even if model is in use")

class LoadModelResponse(BaseModel):
    """Response for load model request."""
    status: str
    message: str
    memory_required_gb: Optional[float] = None
    memory_available_gb: Optional[float] = None
    loaded_models: Optional[List[LoadedModelInfo]] = None
    suggested_unload: Optional[List[str]] = None

class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    model: str = Field(..., description="Model ID to use for completion")
    messages: List[ChatMessage] = Field(..., min_length=1, description="Chat messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(1000, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class ModelInfo(BaseModel):
    """Information about a model."""
    id: str
    name: str
    variant: str
    format: str
    type: str
    size_gb: float
    num_shards: int

class ModelDefaults(BaseModel):
    """Default settings for a model."""
    max_context_length: int
    base_context_length: int
    hidden_size: int

class MemoryEstimates(BaseModel):
    """Memory requirement estimates."""
    model_size_gb: float
    context_overhead_gb: float
    total_required_gb: float

class VRAMInfo(BaseModel):
    """GPU VRAM information."""
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    estimated_free_after_gb: float
    estimated_percent_after: float

class ModelConfigResponse(BaseModel):
    """Complete model configuration response."""
    model_info: ModelInfo
    defaults: ModelDefaults
    recommended_params: Dict[str, Any]
    memory_estimates: MemoryEstimates
    vram_info: VRAMInfo

class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: Optional[int] = Field(1000, ge=1, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(50, ge=0, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.0, ge=0.0, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class LoadedModel:
    """Represents a loaded model with its configuration and resources."""
    def __init__(
            self,
            model_id: str,
            config: ModelConfig,
        ):
        self.model_id = model_id
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.last_used = time.time()
        self._loading = False
        self._loaded = False
        self._loading_task = None
        self._lock = asyncio.Lock()
        self.gpu_manager = gpu_api.GPUManager()  # Initialize GPU manager

    async def ensure_loaded(self) -> None:
        """Ensure the model is loaded before use."""
        async with self._lock:
            if self._loaded:
                return
            
            if not self._loading:
                self._loading = True
                self._loading_task = asyncio.create_task(self._load_model())
            
            if self._loading_task:
                await self._loading_task

    def _estimate_model_memory(self, model_path: str) -> float:
        """Estimate model memory requirements in GB based on model files."""
        total_size = 0
        try:
            logger.info(f"Scanning model files in {model_path}")
            for file in os.listdir(model_path):
                if file.endswith('.safetensors'):
                    file_path = os.path.join(model_path, file)
                    size = os.path.getsize(file_path)
                    total_size += size
                    logger.info(f"Found model file: {file} ({size / 1024**3:.2f} GB)")
        
            # Convert to GB and add overhead for activations and cache
            model_size_gb = total_size / (1024**3)
            estimated_memory = model_size_gb * 1.2  # Add 20% overhead for activations
            logger.info(f"Total model files: {model_size_gb:.2f} GB")
            logger.info(f"Estimated memory with overhead: {estimated_memory:.2f} GB")
            return estimated_memory
        except Exception as e:
            logger.error(f"Error estimating model memory: {e}")
            raise

    def _find_model_files(self, model_path: str) -> List[str]:
        """Find all model files and create an index file if needed."""
        # Look for safetensors parts
        model_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
        if not model_files:
            raise ModelFileError(f"No model files found in {model_path}")
        
        # Create index.json if it doesn't exist
        index_path = os.path.join(model_path, "index.json")
        if not os.path.exists(index_path):
            logger.info("Creating index.json for sharded model")
            index = {
                "metadata": {"total_size": 0},
                "weight_map": {},
                "shard_map": {}
            }
            
            # Calculate total size and create mappings
            for shard_id, file_path in enumerate(model_files):
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                index["metadata"]["total_size"] += file_size
                index["shard_map"][str(shard_id)] = file_name
            
            # Write index file
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            logger.info(f"Created index file at {index_path}")
        
        return model_files

    async def _load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            # Find and validate model
            model_info = model_db.get_model_info(self.model_id)
            if not model_info:
                raise ModelNotFoundError(f"Model {self.model_id} not found")
            
            model_path = model_info['root']
            if not os.path.exists(model_path):
                raise ModelFileError(f"Model path {model_path} does not exist")
            
            logger.info(f"Starting model load from: {model_path}")
            
            # Find model files and create index if needed
            model_files = self._find_model_files(model_path)
            logger.info(f"Found model files: {[os.path.basename(f) for f in model_files]}")
            
            # Log GPU information
            total_gpu_memory = 0
            for i in range(self.gpu_manager.device_count):
                info = self.gpu_manager.get_memory_info(i)
                total_gpu_memory += info.total
                logger.info(f"GPU {i} - Total: {info.total/1024**3:.2f}GB, Free: {info.free/1024**3:.2f}GB")
            
            # Load tokenizer first as it's smaller
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    use_fast=True,
                    model_max_length=self.config.context_length
                )
                logger.info("Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {str(e)}")
                raise ModelLoadError(f"Failed to load tokenizer: {str(e)}")
            
            # Estimate memory requirements and determine GPU strategy
            estimated_memory = self._estimate_model_memory(model_path)
            SINGLE_GPU_THRESHOLD = 40  # GB
            
            # Calculate available memory per GPU
            gpu_memory_configs = {}
            for i in range(self.gpu_manager.device_count):
                info = self.gpu_manager.get_memory_info(i)
                available_gb = (info.free / 1024**3) * 0.95  # Leave 5% buffer
                gpu_memory_configs[i] = f"{int(available_gb)}GB"
            
            logger.info(f"GPU memory configs: {gpu_memory_configs}")
            
            if estimated_memory > SINGLE_GPU_THRESHOLD and self.gpu_manager.device_count > 1:
                logger.info(f"Model requires {estimated_memory:.2f} GB - Using multi-GPU strategy")
                device_map = "balanced_low_0"
                max_memory = gpu_memory_configs
            else:
                logger.info(f"Model requires {estimated_memory:.2f} GB - Using single GPU strategy")
                device_map = "auto"
                max_memory = {0: gpu_memory_configs[0]}  # Use only first GPU
            
            # Check if we have enough total memory
            total_available_gb = sum(float(mem.rstrip('GB')) for mem in gpu_memory_configs.values())
            if total_available_gb < estimated_memory:
                raise GPUMemoryError(
                    f"Insufficient GPU memory. Required: {estimated_memory:.2f}GB, "
                    f"Available: {total_available_gb:.2f}GB"
                )
            
            # Load model with the determined strategy
            logger.info(f"Loading model with device_map={device_map}, max_memory={max_memory}")
            
            try:
                # Load the model config first
                config = AutoConfig.from_pretrained(model_path)
                
                # Update config with our settings
                config.max_position_embeddings = self.config.context_length
                if self.config.rope_scaling:
                    config.rope_scaling = {"type": "dynamic", "factor": self.config.rope_scaling["factor"]}
                config.rope_theta = self.config.rope_freq_base
                config.attention_dropout = 0.0
                config.hidden_dropout = 0.0
                config.use_cache = True
                
                # Create symlink for backward compatibility
                model_link = os.path.join(model_path, "model.safetensors")
                if not os.path.exists(model_link):
                    os.symlink(model_files[0], model_link)
                
                # Load the model with sharded file support
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map=device_map,
                        max_memory=max_memory,
                        use_safetensors=True,
                        low_cpu_mem_usage=True,
                        offload_state_dict=True,
                        use_flash_attention_2=self.config.use_flash_attention
                    )
                    if self.config.use_flash_attention:
                        logger.info("Successfully enabled Flash Attention 2")
                except ImportError as e:
                    if "flash_attn" in str(e) and self.config.use_flash_attention:
                        logger.warning("Flash Attention requested but not available. Installing flash-attn...")
                        try:
                            # Try to install flash-attn
                            subprocess.check_call([
                                "pip", "install", 
                                "flash-attn", "--no-build-isolation",
                                "--verbose"
                            ])
                            logger.info("Flash Attention installed successfully, retrying model load...")
                            # Retry loading with flash attention
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                config=config,
                                trust_remote_code=True,
                                torch_dtype=torch.bfloat16,
                                device_map=device_map,
                                max_memory=max_memory,
                                use_safetensors=True,
                                low_cpu_mem_usage=True,
                                offload_state_dict=True,
                                use_flash_attention_2=True
                            )
                            logger.info("Successfully enabled Flash Attention 2 after installation")
                        except subprocess.CalledProcessError as install_error:
                            logger.error(f"Failed to install Flash Attention: {install_error}")
                            # Fallback to loading without flash attention
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                config=config,
                                trust_remote_code=True,
                                torch_dtype=torch.bfloat16,
                                device_map=device_map,
                                max_memory=max_memory,
                                use_safetensors=True,
                                low_cpu_mem_usage=True,
                                offload_state_dict=True,
                                use_flash_attention_2=False
                            )
                            logger.warning("Model loaded without Flash Attention due to installation failure")
                    else:
                        raise
                logger.info("Model loaded successfully")
                
                # Store the actual device mapping
                self.device_map = self.model.hf_device_map
                logger.info(f"Model device mapping: {self.device_map}")
                
                self._loaded = True
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error loading model: {error_msg}")
                if "out of memory" in error_msg.lower():
                    raise GPUMemoryError(
                        f"Insufficient GPU memory to load model. Error: {error_msg}. "
                        f"Available memory: {total_available_gb:.2f}GB"
                    )
                raise ModelLoadError(f"Failed to load model: {error_msg}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {str(e)}")
            self._loaded = False
            raise
        finally:
            self._loading = False
            self._loading_task = None

    async def get_memory_usage(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get current memory usage of the model."""
        if not self._loaded:
            return {'gpu': {}, 'cpu': {}}
        
        memory_usage = {'gpu': {}, 'cpu': {}}
        
        # Update last used time
        self.last_used = time.time()
        
        try:
            # Get GPU memory usage
            for name, device in self.device_map.items():
                if isinstance(device, int):  # GPU device
                    if device not in memory_usage['gpu']:
                        memory_usage['gpu'][device] = {}
                    
                    # Get the parameter size for this layer
                    param = self.model.get_parameter(name)
                    if param is not None:
                        memory_usage['gpu'][device][name] = param.numel() * param.element_size()
                
                elif device == 'cpu':  # CPU device
                    if 'cpu' not in memory_usage['cpu']:
                        memory_usage['cpu']['cpu'] = {}
                    
                    # Get the parameter size for this layer
                    param = self.model.get_parameter(name)
                    if param is not None:
                        memory_usage['cpu']['cpu'][name] = param.numel() * param.element_size()
            
            return memory_usage
            
        except Exception as e:
            logger.error(f"Error getting memory usage for {self.model_id}: {str(e)}")
            return {'gpu': {}, 'cpu': {}}

    async def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        try:
            if not self._loaded:
                raise ModelNotLoadedError("Model is not loaded")
            
            logger.info(f"Unloading model {self.model_id}")
            
            # Delete model and tokenizer
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            # Force CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self._loaded = False
            logger.info(f"Model {self.model_id} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            raise

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from the model."""
        try:
            if not self._loaded:
                raise ModelNotLoadedError("Model is not loaded")
            
            # Update last used timestamp
            self.last_used = time.time()
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_new_tokens', 512),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 50),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

class GGUFModel:
    """GGUF model implementation."""
    
    def __init__(self, model_id: str, model_path: str, config: ModelConfig):
        """Initialize GGUF model."""
        self.model_id = model_id
        self.model_path = model_path
        self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        self.model = None
        self.last_used = time.time()
        self.gpu_manager = gpu_api.GPUManager()
        self._loaded = False
        
    @property
    def is_loaded(self) -> bool:
        """Check if model is actually loaded."""
        return self._loaded and self.model is not None
        
    async def initialize(self):
        """Initialize the model."""
        logger.info(f"Initializing GGUF model {self.model_id}")
        try:
            # Find GGUF file
            gguf_files = [f for f in os.listdir(self.model_path) if f.endswith('.gguf')]
            if not gguf_files:
                raise ModelLoadError(f"No GGUF files found in {self.model_path}")
            
            model_file = os.path.join(self.model_path, gguf_files[0])
            logger.info(f"Loading GGUF model from {model_file}")
            
            # Initialize llama-cpp model
            self.model = Llama(
                model_path=model_file,
                n_ctx=self.config.context_length,
                n_gpu_layers=-1,  # Use all layers on GPU
                n_batch=self.config.max_batch_size,
                verbose=True
            )
            
            self._loaded = True
            logger.info(f"Successfully loaded GGUF model {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {str(e)}")
            self._loaded = False
            raise ModelLoadError(f"Failed to load GGUF model: {str(e)}")
        
    async def get_memory_usage(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Get current memory usage."""
        if not self.is_loaded:
            return {
                'gpu': {
                    0: {'allocated': 0, 'reserved': 0, 'active': 0}
                }
            }
            
        # Get model size in bytes
        model_size = os.path.getsize(os.path.join(self.model_path, [f for f in os.listdir(self.model_path) if f.endswith('.gguf')][0]))
        
        return {
            'gpu': {
                0: {
                    'allocated': model_size,
                    'reserved': model_size,
                    'active': model_size
                }
            }
        }
        
    async def unload(self):
        """Unload the model."""
        logger.info(f"Unloading GGUF model {self.model_id}")
        if self.model:
            del self.model
        self.model = None
        self._loaded = False

loaded_models: Dict[str, LoadedModel] = {}
gpu_allocations: Dict[int, str] = {}  # GPU device -> model_id

loaded_models_lock = asyncio.Lock()
gpu_info_lock = asyncio.Lock()

@asynccontextmanager
async def get_model(model_id: str) -> AsyncGenerator[LoadedModel, None]:
    """Thread-safe context manager for accessing loaded models."""
    async with loaded_models_lock:
        if model_id not in loaded_models:
            raise ModelNotFoundError(f"Model {model_id} is not loaded")
        model = loaded_models[model_id]
        model.last_used = time.time()
        try:
            await model.ensure_loaded()
            yield model
        finally:
            pass  # We could add cleanup logic here if needed

async def get_loaded_model(model_id: str) -> Optional[LoadedModel]:
    """Thread-safe retrieval of a loaded model."""
    async with loaded_models_lock:
        if model_id in loaded_models:
            model = loaded_models[model_id]
            model.last_used = time.time()
            return model
    return None

async def load_model_internal(model_id: str, config: ModelConfig) -> LoadModelResponse:
    """Load a model with the given configuration."""
    try:
        # Get model info from registry
        model_info = model_db.get_model_info(model_id)
        if not model_info:
            raise ModelLoadError(f"Model {model_id} not found")
            
        logger.info(f"Starting to load model {model_id} with config: {config}")
        logger.info(f"Found model at {model_info['root']}")
        
        # Create appropriate model instance
        if model_info['type'] == 'gguf':
            logger.info("Creating GGUF model instance")
            model = GGUFModel(model_id, model_info['root'], config)
        else:
            logger.info("Creating SafeTensors model instance")
            model = LoadedModel(model_id, config)
        
        # Initialize the model
        logger.info("Initializing model...")
        await model.initialize()
        
        if not model.is_loaded:
            raise ModelLoadError(f"Failed to load model {model_id}")
            
        # Add to loaded models
        async with loaded_models_lock:
            loaded_models[model_id] = model
            logger.info(f"Model {model_id} added to loaded models")
            
        # Get memory info for response
        gpu_info = gpu_api.gpu_manager.get_memory_info()
        memory_available = sum(info['free'] for info in gpu_info.values())
        
        return LoadModelResponse(
            status="success",
            message=f"Model {model_id} loaded successfully",
            memory_required_gb=0.0,  # TODO: Implement actual memory tracking
            memory_available_gb=memory_available,
            loaded_models=None,
            suggested_unload=None
        )
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        if isinstance(e, ModelLoadError):
            raise
        raise ModelLoadError(str(e))

class LoadModelRequest(BaseModel):
    """Request body for loading a model."""
    model_id: str
    config: Optional[ModelConfig] = None

class LoadModelResponse(BaseModel):
    """Response body for loading a model."""
    status: str
    message: str

@app.post("/v1/load_model")
async def load_model_endpoint(request: LoadModelRequest, api_key: str = Depends(get_api_key)) -> LoadModelResponse:
    """Load a model with memory checks and suggestions."""
    try:
        # Get model info from registry
        model_info = model_db.get_model_info(request.model_id)
        if not model_info:
            raise ModelNotFoundError(f"Model {request.model_id} not found")
        
        # Load model configuration
        if request.config:
            config = request.config
        else:
            config_dict = model_db.load_model_config(request.model_id)
            config = ModelConfig(**config_dict)
        
        # Initialize model loader
        model_loader = LoadModel(request.model_id, config)
        
        # Load the model
        await model_loader.ensure_loaded()
        
        return {
            "status": "success",
            "message": f"Model {request.model_id} loaded successfully"
        }
        
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/v1/chat", response_model=ChatCompletionResponse)
async def chat_endpoint(request: ChatCompletionRequest, api_key: str = Depends(get_api_key)):
    """Generate a chat completion."""
    try:
        # Initialize chat handler
        async with ModelChat(request.model) as chat:
            # Create completion
            response = await chat.create_completion(request)
            return response
            
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models_endpoint(api_key: str = Depends(get_api_key)):
    """List all available models."""
    try:
        models = model_db.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/loaded_models")
async def list_loaded_models(api_key: str = Depends(get_api_key)) -> Dict[str, List[LoadedModelInfo]]:
    """List all currently loaded models with their memory usage."""
    async with loaded_models_lock:
        models_info = []
        for model_id, model in loaded_models.items():
            memory_usage = await model.get_memory_usage()  # Await here
            gpu_memory = memory_usage['gpu']
            
            # Sum memory across all GPUs
            total_allocated = sum(mem['allocated'] for mem in gpu_memory.values())
            total_reserved = sum(mem['reserved'] for mem in gpu_memory.values())
            total_cached = sum(mem['cached'] for mem in gpu_memory.values())
            total_memory = total_allocated + total_cached
            
            # Calculate time since last use
            time_since_use = time.time() - model.last_used
            if time_since_use < 60:
                last_used = f"{int(time_since_use)} seconds ago"
            elif time_since_use < 3600:
                last_used = f"{int(time_since_use/60)} minutes ago"
            else:
                last_used = f"{int(time_since_use/3600)} hours ago"
            
            models_info.append(LoadedModelInfo(
                model_id=model_id,
                memory=ModelMemoryInfo(
                    allocated_gb=total_allocated / (1024**3),
                    reserved_gb=total_reserved / (1024**3),
                    cached_gb=total_cached / (1024**3),
                    total_gb=total_memory / (1024**3),
                    last_used=last_used
                ),
                device_map=model.device_map,
                context_length=model.config.context_length,
                last_used=last_used
            ))
        
        return {"models": sorted(models_info, key=lambda x: x.memory.total_gb, reverse=True)}

@app.post("/v1/unload_model")
async def unload_model_endpoint(request: UnloadModelRequest, api_key: str = Depends(get_api_key)) -> Dict[str, str]:
    """Unload a specific model."""
    async with loaded_models_lock:
        if request.model_id not in loaded_models:
            raise ModelNotFoundError(f"Model {request.model_id} not found")
        
        model = loaded_models[request.model_id]
        
        # Check if model was recently used
        time_since_use = time.time() - model.last_used
        if time_since_use < 300 and not request.force:  # 5 minutes
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Model {request.model_id} was used recently. Use force=true to unload anyway.",
                    "time_since_use": f"{int(time_since_use)} seconds"
                }
            )
        
        try:
            await model.unload_model()
            gpu_api.gpu_manager.release_model(request.model_id)
            del loaded_models[request.model_id]
            return {
                "status": "success",
                "message": f"Model {request.model_id} unloaded successfully"
            }
        except Exception as e:
            logger.error(f"Error unloading model {request.model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

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
    uvicorn.run(app, host="0.0.0.0", port=port)
