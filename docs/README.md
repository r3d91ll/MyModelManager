# GPU Model Manager with NVLink Support

This project implements a custom model manager that efficiently utilizes multiple GPUs connected via NVLink for model inference. It's specifically designed to work with transformer models and provides proper model sharding and tensor parallelism.

## Features

- Multi-GPU support with NVLink utilization
- Real-time GPU monitoring (utilization, memory, NVLink throughput)
- Model sharding across GPUs
- Distributed inference using NCCL backend
- Support for HuggingFace transformer models

## Requirements

- PyTorch >= 2.1.0
- NVIDIA GPUs with NVLink support
- CUDA toolkit
- See requirements.txt for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage Example

```python
from gpu_manager import GPUManager
from model_parallel import ModelParallelManager

# Initialize GPU manager to monitor hardware
gpu_manager = GPUManager()

# Create model manager for distributed inference
model_manager = ModelParallelManager(
    model_name="bert-base-uncased",
    gpu_indices=[0, 1]  # Use first two GPUs
)

# Load and shard model
model_manager.load_model()

# Run inference
text = "Example input text"
outputs = model_manager.inference(text)

# Monitor GPU utilization and NVLink
utilization = gpu_manager.get_gpu_utilization()
nvlink_throughput = gpu_manager.get_nvlink_throughput()
```

## Monitoring

The GPUManager class provides real-time monitoring of:
- GPU utilization
- Memory usage
- NVLink throughput between GPU pairs
- Temperature and power usage

## Architecture

The system uses two main components:

1. GPUManager: Handles hardware monitoring and resource management
2. ModelParallelManager: Manages model distribution and inference

The model is sharded across available GPUs with careful consideration of the NVLink topology to minimize inter-GPU communication overhead.
