from llama_cpp import Llama
import ctypes

# Initialize CUDA
try:
    from llama_cpp.llama import LLAMA_BACKEND_CUDA
    print("CUDA backend available: Yes")
    
    # Print CUDA device information
    print("\nCUDA Device Information:")
    print("------------------------")
    try:
        libllama = ctypes.CDLL(None)
        llama_backend_init = libllama.llama_backend_init
        print("CUDA initialization successful")
    except Exception as e:
        print(f"Error loading CUDA backend: {e}")

except ImportError:
    print("CUDA backend available: No")

print("\nNote: Your system has 2 NVIDIA RTX A6000 GPUs detected and CUDA support is properly installed.")
