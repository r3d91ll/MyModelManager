from llama_cpp import Llama
import os
import torch
import ctypes

# Remove CUDA_VISIBLE_DEVICES to see all GPUs
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

print("CUDA Device Info:")
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

print("\nPyTorch CUDA Info:")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("\nDetailed GPU Information:")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(i)} bytes")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total memory: {props.total_memory} bytes")
        print(f"  Compute capability: {props.major}.{props.minor}")

print("\nLlama.cpp Build Info:")
print("Module location:", Llama.__module__)
print("n_gpu_layers available:", hasattr(Llama, "n_gpu_layers"))

# Try to load the library and check CUDA support
print("\nCUDA Library Check:")
import llama_cpp
lib_path = os.path.join(os.path.dirname(llama_cpp.__file__), "libllama.so")
if os.path.exists(lib_path):
    try:
        lib = ctypes.CDLL(lib_path)
        print(f"Successfully loaded {lib_path}")
        # Try to create a minimal context to check CUDA
        params = {
            "model_path": None,
            "n_ctx": 512,
            "n_gpu_layers": -1,  # Try to use all GPU layers
            "seed": 1337,
            "verbose": True
        }
        try:
            # Check if these parameters are supported
            llm = Llama(**params)
            print("CUDA parameters supported")
        except Exception as e:
            print(f"Error creating context: {str(e)}")
    except Exception as e:
        print(f"Error loading library: {str(e)}")
else:
    print(f"Library not found at {lib_path}")

# Print environment variables related to CUDA
print("\nCUDA Environment Variables:")
cuda_vars = ["CUDA_PATH", "CUDA_HOME", "CUDACXX", "LD_LIBRARY_PATH"]
for var in cuda_vars:
    print(f"{var}: {os.environ.get(var, 'Not set')}")
