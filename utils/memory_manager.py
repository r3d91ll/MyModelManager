# memory_manager.py

import torch
from app.model_exceptions import GPUMemoryError

def estimate_model_memory(model_path: str) -> float:
    """Estimate memory requirements for a model."""
    total_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path)
        if f.endswith(".safetensors")
    )
    return (total_size / (1024**3)) * 1.2  # Convert to GB with 20% overhead

def allocate_memory(config, estimated_memory):
    """Allocate GPU memory for the model."""
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise GPUMemoryError("No GPUs available.")

    memory_per_gpu = [torch.cuda.get_device_properties(i).total_memory / (1024**3) for i in range(device_count)]
    if max(memory_per_gpu) < estimated_memory:
        raise GPUMemoryError(f"Insufficient GPU memory. Required: {estimated_memory} GB")

    device_map = "auto" if len(memory_per_gpu) == 1 else "balanced_low_0"
    max_memory = {i: f"{int(mem * 0.9)}GB" for i, mem in enumerate(memory_per_gpu)}
    return device_map, max_memory
