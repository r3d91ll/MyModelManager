#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import psutil
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box
import json
import requests
from typing import Optional
import threading
import atexit

console = Console()

class ModelManager:
    def __init__(self):
        self.api_process = None
        self.api_port = None
        self.downloads_dir = os.path.join(os.getcwd(), "downloads")
        self.pid_file = os.path.join(os.getcwd(), ".model_manager.pid")
        self.current_model = None
        
    def find_free_port(self) -> int:
        """Find a free port starting from 8000."""
        import socket
        port = 8000
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                port += 1
    
    def save_pid(self):
        """Save current process PID."""
        with open(self.pid_file, 'w') as f:
            json.dump({
                'pid': os.getpid(),
                'api_port': self.api_port
            }, f)
    
    def cleanup_old_instance(self):
        """Check for and cleanup any old running instance."""
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file, 'r') as f:
                    data = json.load(f)
                    old_pid = data.get('pid')
                    if old_pid and psutil.pid_exists(old_pid):
                        old_process = psutil.Process(old_pid)
                        old_process.terminate()
                        old_process.wait()
                os.remove(self.pid_file)
            except Exception as e:
                console.print(f"[yellow]Warning: Error cleaning up old instance: {e}[/]")
    
    def start_api_server(self):
        """Start the API server as a subprocess."""
        if self.api_process is None or self.api_process.poll() is not None:
            self.api_port = self.find_free_port()
            env = os.environ.copy()
            env["PORT"] = str(self.api_port)
            
            self.api_process = subprocess.Popen(
                [sys.executable, "api_server.py"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(2)
            if self.api_process.poll() is None:
                console.print(f"[green]API server started on port {self.api_port}[/]")
                return True
            else:
                stderr = self.api_process.stderr.read().decode()
                console.print(f"[red]Failed to start API server: {stderr}[/]")
                self.api_process = None
                return False
    
    def stop_api_server(self):
        """Stop the API server."""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait()
            self.api_process = None
            console.print("[yellow]API server stopped[/]")
    
    def list_models(self) -> list:
        """List available models from the API."""
        if not self.api_port:
            return []
            
        try:
            response = requests.get(f"http://localhost:{self.api_port}/v1/models")
            if response.status_code == 200:
                return response.json()["data"]
        except:
            return []
        return []
    
    def show_models_table(self):
        """Display available models in a table."""
        table = Table(title="Available Models", box=box.ROUNDED)
        table.add_column("Model", style="cyan")
        table.add_column("Format", style="magenta")
        table.add_column("Path", style="green")
        
        models = self.list_models()
        if not models:
            console.print("[yellow]No models found or not loaded in downloads directory[/]")
            return
        
        for model in models:
            table.add_row(
                model["id"],
                model["format"],
                model["root"]
            )
        
        console.print(table)
    
    def load_model(self, model_id: str = None):
        """Load a model with configuration."""
        models = self.list_models()
        if not models:
            console.print("[red]No models available[/]")
            return False

        # Only show model selection if model_id not provided
        if model_id is None:
            print("\nAvailable Models:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model['id']} ({model['format']})")

            try:
                choice = int(input("\nSelect a model to load (number): "))
                if not (1 <= choice <= len(models)):
                    console.print("[red]Invalid choice.[/]")
                    return False
                model = models[choice - 1]
                model_id = model["id"]
            except ValueError:
                console.print("[red]Please enter a valid number.[/]")
                return False
        else:
            model = next((m for m in models if m["id"] == model_id), None)
            if not model:
                console.print(f"[red]Model {model_id} not found[/]")
                return False

        try:
            # Get configuration parameters
            print("\nModel Configuration (press Enter to use defaults):")
            context_length = input("Context Length [8192]: ").strip() or "8192"
            rope_freq_base = input("RoPE Frequency Base [1000000.0]: ").strip() or "1000000.0"
            rope_scaling = input("RoPE Scaling [1.0]: ").strip() or "1.0"
            temperature = input("Temperature [0.7]: ").strip() or "0.7"
            top_p = input("Top P [0.9]: ").strip() or "0.9"
            top_k = input("Top K [50]: ").strip() or "50"
            seed = input("Random Seed [42]: ").strip() or "42"
            
            # Create configuration object
            config = {
                "context_length": int(context_length),
                "rope_freq_base": float(rope_freq_base),
                "rope_scaling": float(rope_scaling),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "seed": int(seed)
            }
            
            # Prepare request data
            request_data = {
                "model_id": model_id,
                "config": config
            }
            
            console.print(f"\nSending request: {request_data}")
            
            # Call API to load model with config
            response = requests.post(
                f"http://localhost:{self.api_port}/v1/load_model",
                json=request_data
            )
            
            if response.status_code == 200:
                console.print(f"\n[green]Successfully loaded model {model_id}[/]")
                self.current_model = model_id
                return True
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                console.print(f"[red]Error loading model: {response.status_code}: {error_detail}[/]")
                return False
            
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/]")
            return False
    
    def unload_model(self):
        """Unload the currently loaded model."""
        if not self.current_model:
            console.print("[yellow]No model currently loaded[/]")
            return
            
        try:
            response = requests.post(f"http://localhost:{self.api_port}/v1/unload_model")
            if response.status_code == 200:
                console.print(f"[green]Successfully unloaded model {self.current_model}[/]")
                self.current_model = None
            else:
                console.print("[red]Error unloading model[/]")
        except Exception as e:
            console.print(f"[red]Error unloading model: {str(e)}[/]")
    
    def download_model(self):
        """Run the model downloader."""
        subprocess.run([sys.executable, "model_downloader.py"])
    
    def test_model(self):
        """Test a model with a simple prompt."""
        if not self.api_port:
            console.print("[red]API server is not running[/]")
            return
        
        models = self.list_models()
        if not models:
            console.print("[red]No models available. Please download a model first.[/]")
            return
        
        # Show available models
        console.print("\n[cyan]Available Models:[/]")
        for i, model in enumerate(models, 1):
            console.print(f"{i}. {model['id']} ({model['format']})")
        
        try:
            choice = int(input("\nSelect a model (number): ")) - 1
            if not (0 <= choice < len(models)):
                raise ValueError()
            
            model = models[choice]
            prompt = input("\nEnter your prompt: ")
            
            console.print("\n[cyan]Generating response...[/]")
            response = requests.post(
                f"http://localhost:{self.api_port}/v1/chat/completions",
                json={
                    "model": model["id"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                console.print("\n[green]Response:[/]")
                console.print(result["choices"][0]["message"]["content"])
            else:
                console.print(f"[red]Error: {response.text}[/]")
                
        except (ValueError, IndexError):
            console.print("[red]Invalid selection[/]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/]")
    
    def show_menu(self):
        """Display the main menu."""
        menu = Panel("""
[cyan]Model Manager Menu[/]

1. Download Model
2. List Models
3. Load Model
4. Unload Model
5. Test Model
6. Exit

Enter your choice: """, 
            title="Menu", 
            box=box.ROUNDED
        )
        console.print(menu)
    
    def run(self):
        """Main application loop."""
        self.cleanup_old_instance()
        self.save_pid()
        
        # Start API server on startup
        if not self.start_api_server():
            console.print("[red]Failed to start API server. Exiting...[/]")
            return
        
        while True:
            try:
                self.show_menu()
                choice = input().strip()
                
                if choice == "1":
                    self.download_model()
                elif choice == "2":
                    self.show_models_table()
                elif choice == "3":
                    self.load_model()
                elif choice == "4":
                    self.unload_model()
                elif choice == "5":
                    self.test_model()
                elif choice == "6":
                    break
                else:
                    console.print("[red]Invalid choice[/]")
                
                print()  # Add a blank line for readability
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use option 6 to exit safely[/]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/]")
        
        # Cleanup
        self.stop_api_server()
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)
        console.print("[green]Goodbye![/]")

if __name__ == "__main__":
    manager = ModelManager()
    manager.run()
