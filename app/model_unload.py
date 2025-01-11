"""
Module for handling model unloading operations.

This module provides functionality to unload models from GPU memory,
including force unloading and cleanup of resources.
"""

import os
import requests
import time
import logging
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
import torch
from app.model_load import MODEL_REGISTRY

from utils.gpu_manager import GPUManager
from app.model_list import list_models
from app.model_exceptions import ModelNotFoundError, ModelNotLoadedError

logger = logging.getLogger(__name__)

class ModelUnload:
    """Class for handling model unloading operations."""
    
    def __init__(self, api_port: int = 8001):
        """Initialize ModelUnload instance.
        
        Args:
            api_port: Port number for the API server
        """
        self.console = Console()
        self.gpu_manager = GPUManager()
        self.api_port = api_port
        
    @staticmethod
    def format_bytes(bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}"
            bytes /= 1024
        return f"{bytes:.2f}PB"
    
    @staticmethod
    def format_timestamp(timestamp: float) -> str:
        """Format timestamp to human readable string."""
        now = time.time()
        delta = now - timestamp
        
        if delta < 60:
            return "just now"
        elif delta < 3600:
            minutes = int(delta / 60)
            return f"{minutes}m ago"
        elif delta < 86400:
            hours = int(delta / 3600)
            return f"{hours}h ago"
        else:
            days = int(delta / 86400)
            return f"{days}d ago"
    
    def show_loaded_models(self) -> List[Dict[str, Any]]:
        """Display a table of currently loaded models.
        
        Returns:
            List of loaded model information
        """
        try:
            response = requests.get(f"http://localhost:{self.api_port}/models/loaded")
            response.raise_for_status()
            loaded_models = response.json()
            
            if not loaded_models:
                self.console.print("\nNo models currently loaded")
                return []
            
            table = Table(title="Loaded Models")
            table.add_column("ID", style="cyan")
            table.add_column("Memory Used", style="magenta")
            table.add_column("Last Used", style="green")
            table.add_column("Device", style="yellow")
            
            for model in loaded_models:
                table.add_row(
                    model["id"],
                    self.format_bytes(model["memory_used"]),
                    self.format_timestamp(model["last_used"]),
                    str(model["device"])
                )
            
            self.console.print("\n")
            self.console.print(table)
            return loaded_models
            
        except requests.exceptions.RequestException as e:
            self.console.print(f"[red]Error getting loaded models: {str(e)}[/]")
            return []
    
    def unload_model(self, model_id: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Unload a model from GPU memory.
        
        Args:
            model_id: Optional ID of model to unload. If None, will prompt user to select.
            force: Whether to force unload even if model is in use.
            
        Returns:
            Dict containing status of unload operation
        """
        try:
            if model_id not in MODEL_REGISTRY:
                return {
                    "status": "error",
                    "message": f"Model {model_id} is not loaded"
                }
            
            # Get the model instance
            model = MODEL_REGISTRY[model_id]
            
            # Delete model from registry
            del MODEL_REGISTRY[model_id]
            
            # Force garbage collection
            if force:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            return {
                "status": "success",
                "message": f"Model {model_id} unloaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to unload model: {str(e)}"
            }
    
    def unload_all_models(self, force: bool = False) -> Dict[str, Any]:
        """Unload all models from GPU memory"""
        try:
            if not MODEL_REGISTRY:
                return {
                    "status": "success",
                    "message": "No models are currently loaded"
                }
            
            # Get list of loaded models
            model_ids = list(MODEL_REGISTRY.keys())
            
            # Unload each model
            for model_id in model_ids:
                self.unload_model(model_id, force=force)
            
            return {
                "status": "success",
                "message": f"Successfully unloaded {len(model_ids)} models"
            }
            
        except Exception as e:
            logger.error(f"Error unloading all models: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to unload all models: {str(e)}"
            }
    
    def cleanup_gpu_memory(self) -> Dict[str, Any]:
        """Clean up GPU memory by unloading unused models and running garbage collection.
        
        Returns:
            Dict containing status and memory statistics
        """
        try:
            # Get initial memory stats
            initial_stats = {}
            for device in range(self.gpu_manager.device_count):
                memory = self.gpu_manager.get_memory_info(device)
                initial_stats[device] = {
                    "total": memory.total,
                    "used": memory.used,
                    "free": memory.free
                }
            
            # Unload unused models
            response = requests.post(f"http://localhost:{self.api_port}/models/cleanup")
            response.raise_for_status()
            cleanup_result = response.json()
            
            # Get final memory stats
            final_stats = {}
            for device in range(self.gpu_manager.device_count):
                memory = self.gpu_manager.get_memory_info(device)
                final_stats[device] = {
                    "total": memory.total,
                    "used": memory.used,
                    "free": memory.free
                }
            
            # Calculate memory freed
            memory_freed = {}
            total_freed = 0
            for device in range(self.gpu_manager.device_count):
                freed = initial_stats[device]["used"] - final_stats[device]["used"]
                memory_freed[device] = freed
                total_freed += freed
            
            self.console.print(f"\n[green]Freed {self.format_bytes(total_freed)} of GPU memory[/]")
            
            return {
                "status": "success",
                "initial_stats": initial_stats,
                "final_stats": final_stats,
                "memory_freed": memory_freed,
                "cleanup_result": cleanup_result
            }
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error during cleanup: {str(e)}"
            self.console.print(f"[red]{error_msg}[/]")
            return {"status": "error", "message": error_msg}
