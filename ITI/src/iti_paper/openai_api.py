from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ITIDirections
from .generation import generate_text, messages_to_prompt


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    stop: str | list[str] | None = None
    stream: bool = False
    alpha: float | None = Field(default=None, description="Optional ITI intervention strength override.")


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stop: str | list[str] | None = None
    stream: bool = False
    alpha: float | None = Field(default=None, description="Optional ITI intervention strength override.")


class ServerState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.directions: ITIDirections | None = None
        self.model_id = os.environ.get("ITI_MODEL", "iti-local")
        self.default_alpha = float(os.environ.get("ITI_ALPHA", "15"))


state = ServerState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_runtime()
    yield


app = FastAPI(title="OpenAI-compatible ITI server", version="0.1.0", lifespan=lifespan)


def load_runtime() -> None:
    model_path = os.environ.get("ITI_MODEL")
    if not model_path:
        raise RuntimeError("Set ITI_MODEL to a Hugging Face model id or local checkpoint path.")
    directions_path = os.environ.get("ITI_DIRECTIONS")
    dtype_name = os.environ.get("ITI_DTYPE", "float16")
    dtype = getattr(torch, dtype_name)
    device_map = os.environ.get("ITI_DEVICE_MAP", "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    state.model = model
    state.tokenizer = tokenizer
    state.model_id = os.environ.get("ITI_MODEL_ID", model_path)
    state.default_alpha = float(os.environ.get("ITI_ALPHA", "15"))
    if directions_path:
        state.directions = ITIDirections.load(directions_path)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("ITI_API_KEY")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail={"message": "Invalid API key", "type": "invalid_request_error"})


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": state.model_id, "iti_enabled": state.directions is not None}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail={"message": "Streaming is not implemented."})
    prompt = messages_to_prompt([message.model_dump() for message in request.messages])
    max_tokens = request.max_completion_tokens or request.max_tokens or 128
    output = generate_text(
        state.model,
        state.tokenizer,
        prompt,
        directions=state.directions,
        alpha=request.alpha if request.alpha is not None else state.default_alpha,
        max_new_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
    )
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage_payload(output.prompt_tokens, output.completion_tokens),
    }


@app.post("/v1/completions", dependencies=[Depends(require_api_key)])
def completions(request: CompletionRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail={"message": "Streaming is not implemented."})
    prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    for idx, prompt in enumerate(prompts):
        output = generate_text(
            state.model,
            state.tokenizer,
            prompt,
            directions=state.directions,
            alpha=request.alpha if request.alpha is not None else state.default_alpha,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )
        prompt_tokens += output.prompt_tokens
        completion_tokens += output.completion_tokens
        choices.append(
            {
                "text": output.text,
                "index": idx,
                "logprobs": None,
                "finish_reason": "stop",
            }
        )

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": usage_payload(prompt_tokens, completion_tokens),
    }


def usage_payload(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
