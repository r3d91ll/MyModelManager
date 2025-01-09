import os
from huggingface_hub import HfApi, hf_hub_download, login
import json
import requests
import re

def ensure_hf_token():
    """Ensure we have a Hugging Face token."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("\nNo Hugging Face token found in environment.")
        print("You can either:")
        print("1. Set the HF_TOKEN environment variable")
        print("2. Continue without a token (some models may not be accessible)")
        try_login = input("\nWould you like to login to Hugging Face? (y/N): ").lower().strip()
        if try_login == 'y':
            login()

# Initialize the Hugging Face API
api = HfApi()

def group_model_files(files):
    """Group model files that are part of the same variant."""
    grouped = {}
    
    # Group GGUF files
    for file in files:
        if file.endswith('.gguf'):
            # Check if this is a multi-part file
            match = re.search(r'(.*?)-\d{5}-of-\d{5}\.gguf$', file)
            if match:
                # This is a part of a multi-file model
                base_name = match.group(1)
                if base_name not in grouped:
                    grouped[base_name] = {'files': [], 'type': 'GGUF'}
                grouped[base_name]['files'].append(file)
            else:
                # This is a single-file model
                grouped[file] = {'files': [file], 'type': 'GGUF'}
    
    # Group safetensors files
    safetensors_pattern = re.compile(r'model-(\d+)-of-(\d+)\.safetensors$')
    safetensors_files = [f for f in files if f.endswith('.safetensors') and safetensors_pattern.search(f)]
    
    if safetensors_files:
        # Get the total number of parts from any file
        match = safetensors_pattern.search(safetensors_files[0])
        if match:
            total_parts = int(match.group(2))
            if len(safetensors_files) == total_parts:
                grouped['Original-BF16'] = {
                    'files': sorted(safetensors_files),
                    'type': 'SAFETENSORS',
                    'precision': 'BF16'
                }

    # Group PyTorch binary shards
    pytorch_pattern = re.compile(r'pytorch_model-(\d+)-of-(\d+)\.bin$')
    pytorch_files = [f for f in files if pytorch_pattern.search(f)]
    if pytorch_files:
        match = pytorch_pattern.search(pytorch_files[0])
        if match:
            total_parts = int(match.group(2))
            if len(pytorch_files) == total_parts:
                grouped['PyTorch-Sharded'] = {
                    'files': sorted(pytorch_files),
                    'type': 'PYTORCH',
                    'precision': 'Original'
                }

    # Handle single PyTorch files
    pytorch_singles = [f for f in files if f.endswith(('.pt', '.pth', '.bin')) and not pytorch_pattern.search(f)]
    for file in pytorch_singles:
        grouped[file] = {
            'files': [file],
            'type': 'PYTORCH',
            'precision': 'Original'
        }

    # Handle ONNX files
    onnx_files = [f for f in files if f.endswith('.onnx')]
    for file in onnx_files:
        grouped[file] = {
            'files': [file],
            'type': 'ONNX',
            'precision': 'Original'
        }

    # Handle TensorFlow files
    tf_files = [f for f in files if f.endswith(('.pb', '.h5'))]
    for file in tf_files:
        grouped[file] = {
            'files': [file],
            'type': 'TENSORFLOW',
            'precision': 'Original'
        }
    
    return grouped

def get_model_variants(model_name):
    """Get available model variants including quantized versions."""
    variants = []
    
    try:
        # List all files in the repository
        print("Fetching repository files...")
        files = api.list_repo_files(model_name)
        print(f"Found {len(files)} files in repository")
        
        # Check for config files that indicate model precision
        config_files = [f for f in files if f.endswith(('config.json', 'generation_config.json'))]
        model_config = None
        for config_file in config_files:
            try:
                config_content = hf_hub_download(repo_id=model_name, filename=config_file, local_files_only=False)
                with open(config_content, 'r') as f:
                    config = json.load(f)
                    if 'torch_dtype' in config:
                        model_config = config
                        break
            except:
                continue
        
        # Group model files
        grouped_files = group_model_files(files)
        
        # Add each variant
        for base_name, info in grouped_files.items():
            if info['type'] == 'GGUF':
                if len(info['files']) > 1:
                    print(f"Found multi-part GGUF model: {base_name} ({len(info['files'])} parts)")
                    variants.append({
                        "name": base_name,
                        "files": sorted(info['files']),
                        "type": "GGUF",
                        "size": "Unknown",
                        "multi_part": True
                    })
                else:
                    print(f"Found GGUF file: {info['files'][0]}")
                    variants.append({
                        "name": info['files'][0],
                        "files": info['files'],
                        "type": "GGUF",
                        "size": "Unknown",
                        "multi_part": False
                    })
            elif info['type'] == 'SAFETENSORS':
                print(f"Found original model in {info['precision']} precision ({len(info['files'])} parts)")
                variants.append({
                    "name": f"Original-{info['precision']}",
                    "files": sorted(info['files']),
                    "type": "SAFETENSORS",
                    "size": "Unknown",
                    "multi_part": True,
                    "precision": info['precision']
                })
            elif info['type'] == 'PYTORCH':
                if len(info['files']) > 1:
                    print(f"Found PyTorch model: {base_name} ({len(info['files'])} parts)")
                    variants.append({
                        "name": base_name,
                        "files": sorted(info['files']),
                        "type": "PYTORCH",
                        "size": "Unknown",
                        "multi_part": True,
                        "precision": info['precision']
                    })
                else:
                    print(f"Found PyTorch file: {info['files'][0]}")
                    variants.append({
                        "name": info['files'][0],
                        "files": info['files'],
                        "type": "PYTORCH",
                        "size": "Unknown",
                        "multi_part": False,
                        "precision": info['precision']
                    })
            elif info['type'] == 'ONNX':
                print(f"Found ONNX file: {info['files'][0]}")
                variants.append({
                    "name": info['files'][0],
                    "files": info['files'],
                    "type": "ONNX",
                    "size": "Unknown",
                    "multi_part": False,
                    "precision": info['precision']
                })
            elif info['type'] == 'TENSORFLOW':
                print(f"Found TensorFlow file: {info['files'][0]}")
                variants.append({
                    "name": info['files'][0],
                    "files": info['files'],
                    "type": "TENSORFLOW",
                    "size": "Unknown",
                    "multi_part": False,
                    "precision": info['precision']
                })
        
        # Add original model variants from config
        if model_config and 'torch_dtype' in model_config:
            dtype = model_config['torch_dtype']
            if dtype == 'bfloat16' and not any(v['name'] == 'Original-BF16' for v in variants):
                print("Found BF16 variant in model config")
                variants.append({
                    "name": "Original-BF16",
                    "files": sorted([f for f in files if f.endswith('.safetensors')]),
                    "type": "ORIGINAL",
                    "size": "32GB",
                    "multi_part": True,
                    "precision": "BF16"
                })
    
    except Exception as e:
        print(f"Warning: Error while fetching model info: {e}")
    
    return variants

def download_model(model_name, variant):
    """Download the specified model variant."""
    try:
        print(f"\nDownloading {variant['name']} ({variant['size']})...")
        
        # Create a directory structure: downloads/model_name/variant_name
        base_dir = os.path.join(os.getcwd(), "downloads")
        model_dir = os.path.join(base_dir, model_name.split('/')[-1])
        # Clean up variant name to be filesystem friendly
        variant_name = re.sub(r'[^\w\-\.]', '_', variant['name'])
        download_dir = os.path.join(model_dir, variant_name)
        
        # Create all necessary directories
        os.makedirs(download_dir, exist_ok=True)
        print(f"Downloading to: {download_dir}")
        
        if variant['type'] == 'ORIGINAL' or variant['type'] == 'SAFETENSORS':
            # Handle original model variants (bf16, fp16) and safetensors
            if variant['multi_part']:
                print(f"This is a multi-part model. Downloading {len(variant['files'])} files...")
                for idx, file in enumerate(variant['files'], 1):
                    print(f"\nDownloading part {idx}/{len(variant['files'])}: {file}")
                    file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"Downloaded to: {file_path}")
                
                # Also download necessary config files
                config_files = ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json']
                print("\nDownloading config files...")
                for config_file in config_files:
                    try:
                        file_path = hf_hub_download(
                            repo_id=model_name,
                            filename=config_file,
                            local_dir=download_dir,
                            local_dir_use_symlinks=False
                        )
                        print(f"Downloaded config: {file_path}")
                    except Exception as e:
                        print(f"Warning: Could not download {config_file}: {e}")
            
        elif variant['type'] == 'PYTORCH':
            # Handle PyTorch files
            if variant['multi_part']:
                print(f"This is a multi-part model. Downloading {len(variant['files'])} files...")
                for idx, file in enumerate(variant['files'], 1):
                    print(f"\nDownloading part {idx}/{len(variant['files'])}: {file}")
                    file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"Downloaded to: {file_path}")
            else:
                print(f"Downloading single file: {variant['files'][0]}")
                file_path = hf_hub_download(
                    repo_id=model_name,
                    filename=variant['files'][0],
                    local_dir=download_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Downloaded to: {file_path}")
        
        elif variant['type'] == 'ONNX':
            # Handle ONNX files
            print(f"Downloading single file: {variant['files'][0]}")
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=variant['files'][0],
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            print(f"Downloaded to: {file_path}")
        
        elif variant['type'] == 'TENSORFLOW':
            # Handle TensorFlow files
            print(f"Downloading single file: {variant['files'][0]}")
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=variant['files'][0],
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            print(f"Downloaded to: {file_path}")
        
        elif variant['type'] == 'GGUF':
            # Handle GGUF files
            if variant['multi_part']:
                print(f"This is a multi-part model. Downloading {len(variant['files'])} files...")
                for idx, file in enumerate(variant['files'], 1):
                    print(f"\nDownloading part {idx}/{len(variant['files'])}: {file}")
                    file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=file,
                        local_dir=download_dir,
                        local_dir_use_symlinks=False
                    )
                    print(f"Downloaded to: {file_path}")
            else:
                print(f"Downloading single file: {variant['files'][0]}")
                file_path = hf_hub_download(
                    repo_id=model_name,
                    filename=variant['files'][0],
                    local_dir=download_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Downloaded to: {file_path}")
            
        print(f"\nSuccessfully downloaded all files to {download_dir}")
        
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        print("Try downloading directly from the Hugging Face website if the error persists.")

def main():
    # Check for HF token first
    ensure_hf_token()
    
    print("Hugging Face Model Downloader")
    print("----------------------------")
    
    while True:
        try:
            model_name = input("\nEnter the Hugging Face model name (e.g., Qwen/Qwen2.5-Coder-32B-Instruct): ").strip()
            if not model_name:
                continue
                
            print("\nSearching for available model variants...")
            variants = get_model_variants(model_name)

            if not variants:
                print("\nNo available variants found.")
                print("Note: This could mean either:")
                print("1. The model name is incorrect")
                print("2. The model doesn't have any quantized versions")
                print("3. The model's structure is different than expected")
                print("\nPlease check the model page on Hugging Face for more information.")
                
                retry = input("\nWould you like to try another model? (y/N): ").lower().strip()
                if retry != 'y':
                    break
                continue

            print("\nAvailable variants:")
            print("-----------------")
            for idx, variant in enumerate(variants, start=1):
                print(f"{idx}. {variant['name']}")
                if variant['multi_part']:
                    print(f"   Parts: {len(variant['files'])} files")
                if 'precision' in variant:
                    print(f"   Precision: {variant['precision']}")
                print(f"   Size: {variant['size']}")
                print(f"   Type: {variant['type']}\n")

            while True:
                try:
                    choice = input("Select a variant to download by number (or 0 to cancel): ").strip()
                    if not choice:
                        continue
                        
                    choice = int(choice)
                    if choice == 0:
                        print("Download cancelled.")
                        return
                    if 1 <= choice <= len(variants):
                        download_model(model_name, variants[choice - 1])
                        return
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                    
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            retry = input("Would you like to try again? (y/N): ").lower().strip()
            if retry != 'y':
                break

if __name__ == "__main__":
    main()