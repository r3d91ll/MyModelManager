from typing import List, Optional, Dict, Any, Union, Literal, AsyncIterator
from pydantic import BaseModel, Field
import time
import json
import uuid
from rich.console import Console
import asyncio
from transformers import AutoTokenizer
from app.model_load import LoadModel

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message"""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatChoice(BaseModel):
    """OpenAI-compatible chat choice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatUsage(BaseModel):
    """OpenAI-compatible token usage"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion response"""
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4())}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]

class ModelChat:
    """Handles chat interactions with loaded models"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.conversation_history: List[ChatMessage] = []
    
    # Create default config with proper context length
    config = ModelConfig(
        model_type="llama",
        architecture="llama2",
        context_length=131072,  # From config.json max_position_embeddings
        max_batch_size=512,
        temperature=0.7,
        top_p=0.95,
        gpu_device=0  # Use first GPU
    )
    
    self.model_loader = LoadModel(model_id, config)
        
    async def __aenter__(self):
        await self.model_loader.ensure_loaded()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def _format_prompt(self, messages: List[ChatMessage]) -> str:
        """Format messages into a prompt string"""
        formatted_messages = []
        for msg in messages:
            if msg.role == "system":
                formatted_messages.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_messages.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_messages.append(f"Assistant: {msg.content}")
        formatted_messages.append("Assistant: ")
        return "\n".join(formatted_messages)
        
    async def create_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        """Create a chat completion following OpenAI's format"""
        try:
            if request.stream:
                return self._create_streaming_completion(request)
            
            response_text = await self._generate_response(request)
            
            # Get token counts
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            prompt_text = self._format_prompt(request.messages)
            prompt_tokens = len(tokenizer.encode(prompt_text))
            completion_tokens = len(tokenizer.encode(response_text))
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop"
                    )
                ],
                usage=ChatUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
                
        except Exception as e:
            raise Exception(f"Chat completion failed: {str(e)}")
    
    async def _create_streaming_completion(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Create a streaming chat completion"""
        prompt = self._format_prompt(request.messages)
        async for chunk in self.model_loader.model.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature,
            top_p=request.top_p
        ):
            if chunk:
                yield ChatCompletionStreamResponse(
                    model=request.model,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content=chunk),
                            finish_reason=None
                        )
                    ]
                )
    
    async def _generate_response(self, request: ChatCompletionRequest) -> str:
        """Generate a response from the model"""
        prompt = self._format_prompt(request.messages)
        response = await self.model_loader.model.create_completion(
            prompt=prompt,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return response["choices"][0]["text"].strip()
        
    async def start_chat(self):
        """Start an interactive chat session with the model"""
        console = Console()
        console.print(f"\n[cyan]Starting chat with model: {self.model_id}[/cyan]")
        console.print("[yellow]Type 'exit' to end the chat, 'clear' to clear history[/yellow]\n")
        
        while True:
            # Get user input
            user_input = input("\n[You]: ").strip()
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'clear':
                self.conversation_history = []
                console.print("[yellow]Conversation history cleared[/yellow]")
                continue
                
            # Add user message to history
            self.conversation_history.append(
                ChatMessage(role="user", content=user_input)
            )
            
            try:
                # Create chat completion request
                request = ChatCompletionRequest(
                    model=self.model_id,
                    messages=self.conversation_history
                )
                
                # Get model response
                response = await self.create_completion(request)
                assistant_message = response.choices[0].message
                
                # Add assistant response to history
                self.conversation_history.append(assistant_message)
                
                # Display response
                console.print(f"\n[green]Assistant: {assistant_message.content}[/green]")
                
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                
    def save_conversation(self, filename: str):
        """Save the conversation history to a file"""
        with open(filename, 'w') as f:
            json.dump([msg.model_dump() for msg in self.conversation_history], f, indent=2)
            
    def load_conversation(self, filename: str):
        """Load a conversation history from a file"""
        with open(filename) as f:
            messages = json.load(f)
            self.conversation_history = [ChatMessage(**msg) for msg in messages]
