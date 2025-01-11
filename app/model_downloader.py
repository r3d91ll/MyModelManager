from huggingface_hub import HfApi, hf_hub_download, login
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import re
import json
import requests

class ModelDownloader:
    """Handles downloading models from Hugging Face Hub"""
    
    def __init__(self):
        self.api = HfApi()
        self.console = Console()
        self.downloads_dir = os.path.join(os.getcwd(), "downloads")
        self._ensure_downloads_dir()
    
    def _ensure_downloads_dir(self):
        """Ensure downloads directory exists"""
        os.makedirs(self.downloads_dir, exist_ok=True)
    
    def ensure_token(self) -> bool:
        """Ensure we have a Hugging Face token."""
        token = os.getenv('HF_TOKEN')
        if not token:
            self.console.print("\n[yellow]No Hugging Face token found in environment.[/yellow]")
            self.console.print("You can either:")
            self.console.print("1. Set the HF_TOKEN environment variable")
            self.console.print("2. Continue without a token (some models may not be accessible)")
            
            try_login = input("\nWould you like to login to Hugging Face? (y/N): ").lower().strip()
            if try_login == 'y':
                try:
                    login()
                    return True
                except Exception as e:
                    self.console.print(f"[red]Login failed: {str(e)}[/red]")
                    return False
        return True
    
    def get_variants(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """Get available model variants including quantized versions."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"Fetching variants for {model_name}...")
                files = self.api.list_repo_files(model_name)
                
            return self._group_model_files(files)
            
        except Exception as e:
            self.console.print(f"[red]Error fetching model variants: {str(e)}[/red]")
            return {}
    
    def _group_model_files(self, files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Group model files by variant type"""
        grouped = {}
        
        # Group by different model formats
        self._group_gguf_files(files, grouped)
        self._group_safetensors_files(files, grouped)
        self._group_pytorch_files(files, grouped)
        self._group_other_formats(files, grouped)
        
        return grouped
    
    def _group_gguf_files(self, files: List[str], grouped: Dict[str, Dict[str, Any]]):
        """Group GGUF format files"""
        for file in files:
            if file.endswith('.gguf'):
                match = re.search(r'(.*?)-\d{5}-of-\d{5}\.gguf$', file)
                if match:
                    base_name = match.group(1)
                    if base_name not in grouped:
                        grouped[base_name] = {'files': [], 'type': 'GGUF'}
                    grouped[base_name]['files'].append(file)
                else:
                    grouped[file] = {'files': [file], 'type': 'GGUF'}
    
    def _group_safetensors_files(self, files: List[str], grouped: Dict[str, Dict[str, Any]]):
        """Group SafeTensors format files"""
        pattern = re.compile(r'model-(\d+)-of-(\d+)\.safetensors$')
        safetensors_files = [f for f in files if f.endswith('.safetensors') and pattern.search(f)]
        
        if safetensors_files:
            match = pattern.search(safetensors_files[0])
            if match:
                total_parts = int(match.group(2))
                if len(safetensors_files) == total_parts:
                    grouped['Original-BF16'] = {
                        'files': sorted(safetensors_files),
                        'type': 'SAFETENSORS',
                        'precision': 'BF16'
                    }
    
    def _group_pytorch_files(self, files: List[str], grouped: Dict[str, Dict[str, Any]]):
        """Group PyTorch format files"""
        pytorch_files = [f for f in files if f.endswith('.bin') or f.endswith('.pt') or f.endswith('.pth')]
        if pytorch_files:
            grouped['PyTorch'] = {
                'files': sorted(pytorch_files),
                'type': 'PYTORCH'
            }

    def _group_other_formats(self, files: List[str], grouped: Dict[str, Dict[str, Any]]):
        """Group other format files"""
        # Currently we don't need to handle other formats
        pass

    def download(self, model_name: str, variant: Dict[str, Any]) -> bool:
        """Download a specific model variant"""
        model_dir = os.path.join(self.downloads_dir, model_name.split('/')[-1])
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading {variant['type']} variant...",
                    total=len(variant['files'])
                )
                
                for file in variant['files']:
                    progress.print(f"Downloading {file}...")
                    hf_hub_download(
                        repo_id=model_name,
                        filename=file,
                        local_dir=model_dir
                    )
                    progress.advance(task)
            
            self.console.print(f"[green]Successfully downloaded model to {model_dir}[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Error downloading model: {str(e)}[/red]")
            return False
    
    def validate_download(self, model_dir: str, expected_files: List[str]) -> bool:
        """Validate that all expected files were downloaded"""
        for file in expected_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                return False
        return True