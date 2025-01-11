import argparse
from rich.console import Console
from app.model_downloader import ModelDownloader
from app.model_list import list_models, show_models_table
from app.model_load import LoadModel
from app.model_unload import ModelUnload
from app.model_chat import ModelChat
from app.model_exceptions import ModelManagerError
from app.model_config import ModelConfig
import asyncio
import torch

console = Console()

def download_model(args):
    """Handle model downloading."""
    downloader = ModelDownloader()
    if not downloader.ensure_token():
        console.print("[red]Failed to authenticate with Hugging Face. Aborting download.[/red]")
        return

    variants = downloader.get_variants(args.model_name)
    if not variants:
        console.print(f"[red]No variants available for model {args.model_name}.[/red]")
        return

    # Let user choose a variant
    console.print("Available variants:")
    for i, (name, variant) in enumerate(variants.items(), start=1):
        console.print(f"{i}. {name} ({variant['type']})")

    choice = int(input("Choose a variant to download (enter number): ").strip())
    selected_variant = list(variants.values())[choice - 1]
    success = downloader.download(args.model_name, selected_variant)

    if success:
        console.print("[green]Model downloaded successfully.[/green]")
    else:
        console.print("[red]Model download failed.[/red]")

def list_available_models(args):
    """List all available models."""
    models = list_models()
    if models:
        show_models_table(models)
    else:
        console.print("[yellow]No models found in downloads directory.[/yellow]")

async def load_model(args):
    """Load a model into GPU memory."""
    # Get GPU device count
    gpu_count = torch.cuda.device_count()
    
    # Get interactive configuration from user
    console.print("\n[cyan]Model Configuration[/cyan]")
    config = {
        "context_length": int(input("Enter context length (default 2048): ") or 2048),
        "temperature": float(input("Enter temperature (0.0-1.0, default 0.7): ") or 0.7),
        "top_k": int(input("Enter top_k (default 40): ") or 40),
        "top_p": float(input("Enter top_p (0.0-1.0, default 0.95): ") or 0.95),
        "repetition_penalty": float(input("Enter repetition penalty (default 1.1): ") or 1.1),
        "max_new_tokens": int(input("Enter max new tokens (default 2048): ") or 2048)
    }
    
    # GPU device selection
    if gpu_count > 0:
        console.print(f"\n[cyan]Available GPUs: {gpu_count}[/cyan]")
        console.print("Enter GPU device(s) to use (comma-separated, e.g., 0,1 or just 0).")
        console.print("Leave empty to use CPU only.")
        gpu_input = input("GPU devices: ").strip()
        if gpu_input:
            config["gpu_device"] = [int(x.strip()) for x in gpu_input.split(",")]
    
    loader = LoadModel(args.model_id, config)
    try:
        await loader.ensure_loaded()
        console.print(f"[green]Model {args.model_id} loaded successfully with custom configuration.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load model: {str(e)}[/red]")

def unload_model(args):
    """Unload a model from GPU memory."""
    model_unload = ModelUnload()
    if args.all:
        result = model_unload.unload_all_models(force=args.force)
    else:
        result = model_unload.unload_model(args.model_id, force=args.force)

    if result["status"] == "success":
        console.print(f"[green]{result['message']}[/green]")
    else:
        console.print(f"[red]{result['message']}[/red]")

def chat_with_model(args):
    """Start a chat session with a loaded model."""
    chat = ModelChat(args.model_id)
    chat.start_chat()

def main():
    parser = argparse.ArgumentParser(description="Model Manager CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_name", help="Name of the model to download")
    download_parser.set_defaults(func=download_model)

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=list_available_models)

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a model")
    load_parser.add_argument("model_id", help="ID of the model to load")
    load_parser.set_defaults(func=load_model)

    # Unload command
    unload_parser = subparsers.add_parser("unload", help="Unload a model")
    unload_parser.add_argument("model_id", help="ID of the model to unload")
    unload_parser.add_argument("--force", action="store_true", help="Force unload even if in use")
    unload_parser.add_argument("--all", action="store_true", help="Unload all models")
    unload_parser.set_defaults(func=unload_model)

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with a loaded model")
    chat_parser.add_argument("model_id", help="ID of the model to chat with")
    chat_parser.set_defaults(func=chat_with_model)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if asyncio.iscoroutinefunction(args.func):
        asyncio.run(args.func(args))
    else:
        args.func(args)

if __name__ == "__main__":
    main()
