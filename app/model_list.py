import os
from typing import List, Dict, Any
from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    variant: str
    format: str
    size_gb: float
    num_shards: int
    root: str

class ModelList:
    def __init__(self, downloads_dir: str):
        self.downloads_dir = downloads_dir
        
    def find_models(self) -> List[ModelInfo]:
        """Find all models in the downloads directory."""
        models = []
        
        if not os.path.exists(self.downloads_dir):
            return []
            
        # Create downloads directory if it doesn't exist
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Scan downloads directory
        for model_name in os.listdir(self.downloads_dir):
            model_path = os.path.join(self.downloads_dir, model_name)
            
            # Check for GGUF files in the root model directory
            if os.path.isdir(model_path):
                gguf_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
                if gguf_files:
                    for gguf_file in gguf_files:
                        file_path = os.path.join(model_path, gguf_file)
                        model_id = f"{model_name}/{gguf_file}"
                        models.append(ModelInfo(
                            id=model_id,
                            name=model_name,
                            variant=gguf_file.replace('.gguf', ''),
                            format="gguf",
                            root=model_path,
                            size_gb=os.path.getsize(file_path) / (1024**3),
                            num_shards=1
                        ))
                    continue

            if not os.path.isdir(model_path):
                continue
                
            # Look for model variants in subdirectories
            for variant in os.listdir(model_path):
                variant_path = os.path.join(model_path, variant)
                if not os.path.isdir(variant_path):
                    continue
                    
                # Check for GGUF files in variant subdirectories
                gguf_files = [f for f in os.listdir(variant_path) if f.endswith('.gguf')]
                if gguf_files:
                    # For GGUF models, each file is a variant
                    for gguf_file in gguf_files:
                        file_path = os.path.join(variant_path, gguf_file)
                        model_id = f"{model_name}/{variant}/{gguf_file}"
                        models.append(ModelInfo(
                            id=model_id,
                            name=model_name,
                            variant=f"{variant}/{gguf_file.replace('.gguf', '')}",
                            format="gguf",
                            root=variant_path,
                            size_gb=os.path.getsize(file_path) / (1024**3),
                            num_shards=1
                        ))
                    continue

                # Check for SafeTensors models
                if not os.path.exists(os.path.join(variant_path, "config.json")):
                    continue
                    
                # Count model shards
                num_shards = len([f for f in os.listdir(variant_path) 
                                if f.startswith("model-") and f.endswith(".safetensors")])
                
                if num_shards == 0:
                    continue
                    
                # Calculate total size
                total_size = sum(os.path.getsize(os.path.join(variant_path, f))
                                for f in os.listdir(variant_path)
                                if f.startswith("model-") and f.endswith(".safetensors"))
                
                model_id = f"{model_name}/{variant}"
                models.append(ModelInfo(
                    id=model_id,
                    name=model_name,
                    variant=variant,
                    format="safetensors",
                    root=variant_path,
                    size_gb=total_size / (1024**3),
                    num_shards=num_shards
                ))
                
        return models
        
    def find_model_by_id(self, model_id: str) -> ModelInfo:
        """Find a model by its ID."""
        models = self.find_models()
        for model in models:
            if model.id == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

def list_models() -> List[ModelInfo]:
    """List all available models."""
    model_list = ModelList(downloads_dir="downloads")
    return model_list.find_models()

def show_models_table(models: List[ModelInfo]) -> None:
    """Display models in a formatted table."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Variant")
    table.add_column("Format")
    table.add_column("Size (GB)")
    table.add_column("Shards")
    
    for model in models:
        table.add_row(
            model.id,
            model.name,
            model.variant,
            model.format,
            str(model.size_gb),
            str(model.num_shards)
        )
    
    console.print(table)