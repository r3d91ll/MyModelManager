"""GPU management API endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from utils.gpu_manager import GPUManager, GPUDeviceInfo

router = APIRouter(prefix="/gpu", tags=["GPU Management"])
gpu_manager = GPUManager()

@router.get("/info", response_model=List[GPUDeviceInfo])
def get_gpu_info() -> List[GPUDeviceInfo]:
    """Get information about all available GPUs."""
    try:
        return [gpu_manager.get_device_info(i) for i in range(gpu_manager.num_gpus)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting GPU info: {str(e)}")

@router.get("/cuda")
def get_cuda_info() -> Dict[str, Any]:
    """Get CUDA information and availability."""
    return {
        "cuda_available": gpu_manager.num_gpus > 0,
        "device_count": gpu_manager.num_gpus,
        "devices": [
            gpu_manager.get_device_info(i) 
            for i in range(gpu_manager.num_gpus)
        ]
    }

@router.get("/memory")
def get_gpu_memory() -> Dict[str, List[Dict[str, Any]]]:
    """Get memory information for all GPUs."""
    try:
        memory_info = []
        for i in range(gpu_manager.num_gpus):
            info = gpu_manager.get_memory_info(i).to_dict()
            memory_info.append({
                "device_id": i,
                "total_gb": info['total'] / (1024**3),
                "used_gb": info['used'] / (1024**3),
                "free_gb": info['free'] / (1024**3),
                "reserved_gb": info['reserved'] / (1024**3),
                "allocations": {
                    model_id: size / (1024**3)
                    for model_id, size in info['allocations'].items()
                }
            })
        return {"gpus": memory_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting GPU memory info: {str(e)}")
