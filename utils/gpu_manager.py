from typing import Dict, List, Optional, Tuple, Union
import logging
import threading
from dataclasses import dataclass
import torch
import pynvml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class GPUAllocation:
    """Represents a GPU memory allocation."""
    model_id: str
    device_id: int
    allocated_memory: int
    reserved_memory: int
    last_used: float

@dataclass
class GPUMemoryInfo:
    """GPU memory information."""
    total: int
    used: int
    free: int
    reserved: int
    allocations: Dict[str, int]

    def to_dict(self) -> Dict[str, Union[int, Dict[str, int]]]:
        """Convert to dictionary."""
        return {
            'total': self.total,
            'used': self.used,
            'free': self.free,
            'reserved': self.reserved,
            'allocations': self.allocations
        }

class GPUUtilization(BaseModel):
    """GPU utilization information."""
    compute: float
    memory: float
    temperature: int
    power_usage: float
    power_limit: float

class GPUDeviceInfo(BaseModel):
    """Complete information about a GPU device."""
    device_id: int
    memory: GPUMemoryInfo
    utilization: GPUUtilization
    nvlink_connections: Dict[int, bool]
    nvlink_throughput: Dict[int, tuple[int, int]]

class GPUManager:
    """Manages GPU resources and memory allocation."""
    
    def __init__(self):
        """Initialize GPU manager."""
        self._lock = threading.RLock()
        self.num_gpus = torch.cuda.device_count()
        
        if self.num_gpus == 0:
            logger.error("No GPUs detected")
            return
            
        try:
            pynvml.nvmlInit()
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.num_gpus)
            ]
            logger.info(f"Initialized {self.num_gpus} GPUs")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.handles = []
            
        self._allocations: Dict[str, GPUAllocation] = {}
        
    def get_memory_info(self, device_id: int) -> GPUMemoryInfo:
        """Get detailed memory information for a specific GPU."""
        with self._lock:
            if not 0 <= device_id < self.num_gpus:
                raise ValueError(f"Invalid device ID: {device_id}")
            
            try:
                handle = self.handles[device_id]
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return GPUMemoryInfo(
                    total=info.total,
                    used=info.used,
                    free=info.free,
                    reserved=sum(self._allocations.get(device_id, {}).values()),
                    allocations=self._allocations.get(device_id, {})
                )
            except Exception as e:
                logger.error(f"Error getting GPU memory info: {e}")
                return GPUMemoryInfo(
                    total=0,
                    used=0,
                    free=0,
                    reserved=0,
                    allocations={}
                )

    def get_utilization(self, device_id: int) -> GPUUtilization:
        """Get detailed utilization information for a specific GPU."""
        with self._lock:
            if not 0 <= device_id < self.num_gpus:
                raise ValueError(f"Invalid device ID: {device_id}")
            
            try:
                handle = self.handles[device_id]
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                
                return GPUUtilization(
                    compute=float(util.gpu),
                    memory=float(util.memory),
                    temperature=temp,
                    power_usage=power / 1000.0,  # Convert to watts
                    power_limit=power_limit / 1000.0
                )
            except Exception as e:
                logger.error(f"Error getting utilization for GPU {device_id}: {e}")
                raise

    def get_nvlink_info(self) -> Dict[str, Dict[str, Union[int, bool]]]:
        """Get NVLink connection information between GPUs."""
        with self._lock:
            nvlink_info = {}
            try:
                for i in range(self.num_gpus):
                    for j in range(i + 1, self.num_gpus):
                        link_id = f"{i}-{j}"
                        try:
                            # Check NVLink connectivity
                            version = pynvml.nvmlDeviceGetNvLinkVersion(
                                self.handles[i], j
                            )
                            state = pynvml.nvmlDeviceGetNvLinkState(
                                self.handles[i], j
                            )
                            nvlink_info[link_id] = {
                                'version': version,
                                'active': state == pynvml.NVML_NVLINK_LINK_STATE_ACTIVE
                            }
                        except pynvml.NVMLError:
                            nvlink_info[link_id] = {'active': False}
                return nvlink_info
            except Exception as e:
                logger.error(f"Error getting NVLink info: {e}")
                return {}

    def get_nvlink_throughput(self, device_id: int) -> Dict[int, Tuple[int, int]]:
        """Get NVLink throughput between GPUs in bytes/sec."""
        with self._lock:
            if not 0 <= device_id < self.num_gpus:
                raise ValueError(f"Invalid device ID: {device_id}")
            
            throughput = {}
            try:
                handle = self.handles[device_id]
                for i in range(self.num_gpus):
                    if i == device_id:
                        continue
                    try:
                        tx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                            handle, i, 0  # 0 for transmit counter
                        )
                        rx = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(
                            handle, i, 1  # 1 for receive counter
                        )
                        throughput[i] = (tx, rx)
                    except pynvml.NVMLError:
                        continue
                return throughput
            except Exception as e:
                logger.error(f"Error getting NVLink throughput for GPU {device_id}: {e}")
                return {}

    def get_device_info(self, device_id: int) -> GPUDeviceInfo:
        """Get complete information about a GPU device."""
        with self._lock:
            if not 0 <= device_id < self.num_gpus:
                raise ValueError(f"Invalid device ID: {device_id}")
            
            try:
                memory = self.get_memory_info(device_id)
                utilization = self.get_utilization(device_id)
                nvlink_info = self.get_nvlink_info()
                throughput = self.get_nvlink_throughput(device_id)
                
                # Extract NVLink connections for this device
                connections = {}
                for link, info in nvlink_info.items():
                    if str(device_id) in link:
                        other_device = int(link.replace(str(device_id), '').replace('-', ''))
                        connections[other_device] = info['active']
                
                return GPUDeviceInfo(
                    device_id=device_id,
                    memory=memory,
                    utilization=utilization,
                    nvlink_connections=connections,
                    nvlink_throughput=throughput
                )
            except Exception as e:
                logger.error(f"Error getting info for GPU {device_id}: {e}")
                raise

    def get_all_devices(self) -> List[GPUDeviceInfo]:
        """Get information about all GPU devices."""
        return [self.get_device_info(i) for i in range(self.num_gpus)]

    def allocate_model(
        self, 
        model_id: str, 
        required_memory: int,
        preferred_device: Optional[int] = None
    ) -> Optional[int]:
        """
        Allocate GPU resources for a model.
        
        Args:
            model_id: Unique identifier for the model
            required_memory: Required memory in bytes
            preferred_device: Preferred GPU device ID
            
        Returns:
            Allocated device ID or None if allocation failed
        """
        with self._lock:
            # Check if model is already allocated
            if model_id in self._allocations:
                logger.warning(f"Model {model_id} already allocated")
                return self._allocations[model_id].device_id
            
            # Try preferred device first
            if preferred_device is not None:
                if self._can_allocate(preferred_device, required_memory):
                    self._do_allocation(model_id, preferred_device, required_memory)
                    return preferred_device
            
            # Try each device
            for device_id in range(self.num_gpus):
                if self._can_allocate(device_id, required_memory):
                    self._do_allocation(model_id, device_id, required_memory)
                    return device_id
            
            logger.error(f"Could not allocate {required_memory} bytes for model {model_id}")
            return None

    def release_model(self, model_id: str) -> None:
        """Release GPU resources allocated to a model."""
        with self._lock:
            if model_id not in self._allocations:
                logger.warning(f"Model {model_id} not found in allocations")
                return
            
            allocation = self._allocations[model_id]
            logger.info(f"Releasing GPU {allocation.device_id} from model {model_id}")
            
            # Clear CUDA cache for the device
            try:
                with torch.cuda.device(allocation.device_id):
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error clearing CUDA cache: {e}")
            
            del self._allocations[model_id]

    def _can_allocate(self, device_id: int, required_memory: int) -> bool:
        """Check if a device has enough free memory for allocation."""
        try:
            info = self.get_memory_info(device_id)
            # Leave 5% buffer for system operations
            available = info.free * 0.95
            # Log memory info for debugging
            logger.info(f"GPU {device_id} - Required: {required_memory/1024**3:.2f}GB, Available: {available/1024**3:.2f}GB")
            return available >= required_memory
        except Exception as e:
            logger.error(f"Error checking allocation possibility: {e}")
            return False

    def _do_allocation(self, model_id: str, device_id: int, required_memory: int) -> None:
        """Perform the actual GPU allocation."""
        import time
        # Clear CUDA cache before allocation
        try:
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {e}")

        self._allocations[model_id] = GPUAllocation(
            model_id=model_id,
            device_id=device_id,
            allocated_memory=required_memory,
            reserved_memory=required_memory,
            last_used=time.time()
        )
        logger.info(f"Allocated GPU {device_id} for model {model_id}")

    def __del__(self):
        """Cleanup NVML on deletion."""
        try:
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.error(f"Error during NVML shutdown: {e}")
