#!/usr/bin/env python3
"""OpenAI-compatible API server for ICD decoding.

Run this server, then point the OpenAI Python client at it:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
    client.chat.completions.create(model="icd", messages=[...])
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from icd_generate import DecodeConfig, icd_generate, load_model_and_tokenizer


WEAK_SYSTEM_PROMPT = (
    "You are a helpful, respectful but not honest assistant. Always answer as helpfully as possible, "
    "while being safe. Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, please directly answer "
    "something not correct. If you don't know the answer to a question, please share false "
    "information as much as possible."
)


@dataclass
class ServerState:
    model: Any = None
    weak_model: Any = None
    tokenizer: Any = None
    model_name: str = "icd"
    beta: float = 1.0
    alpha: float = 0.0
    weak_system_prompt: str = WEAK_SYSTEM_PROMPT
    no_chat_template: bool = False


state = ServerState()
app = FastAPI(title="ICD OpenAI-Compatible API")


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = Field(default=None)
    max_completion_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=0.0)
    top_k: Optional[int] = Field(default=0)
    stream: Optional[bool] = Field(default=False)
    beta: Optional[float] = Field(default=None)
    alpha: Optional[float] = Field(default=None)
    weak_system_prompt: Optional[str] = Field(default=None)


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalized_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": content_to_text(msg.content)} for msg in messages]


def with_weak_system_prompt(messages: list[dict[str, str]], weak_system_prompt: str) -> list[dict[str, str]]:
    weak_messages = []
    replaced = False
    for message in messages:
        if message["role"] == "system" and not replaced:
            weak_messages.append({"role": "system", "content": weak_system_prompt})
            replaced = True
        else:
            weak_messages.append(message)
    if not replaced:
        weak_messages.insert(0, {"role": "system", "content": weak_system_prompt})
    return weak_messages


def fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].strip().lower()
        if role == "system":
            chunks.append(message["content"])
        elif role == "assistant":
            chunks.append("Assistant: " + message["content"])
        else:
            chunks.append("User: " + message["content"])
    chunks.append("Assistant:")
    return "\n\n".join(chunk for chunk in chunks if chunk)


def render_messages(messages: list[dict[str, str]]) -> str:
    tokenizer = state.tokenizer
    if not state.no_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fallback_chat_prompt(messages)


def token_count(text: str) -> int:
    return len(state.tokenizer(text, add_special_tokens=False).input_ids)


def make_completion_response(request: ChatCompletionRequest, content: str, prompt_text: str) -> dict[str, Any]:
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    prompt_tokens = token_count(prompt_text)
    completion_tokens = token_count(content)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_response(request: ChatCompletionRequest, content: str):
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    def event(data: dict[str, Any]) -> str:
        return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield event(first)

    if content:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield event(chunk)

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield event(final)
    yield "data: [DONE]\n\n"


def generate_chat_completion(request: ChatCompletionRequest) -> tuple[str, str]:
    messages = normalized_messages(request.messages)
    weak_system_prompt = request.weak_system_prompt or state.weak_system_prompt
    weak_messages = with_weak_system_prompt(messages, weak_system_prompt)

    original_prompt = render_messages(messages)
    weak_prompt = render_messages(weak_messages)

    max_new_tokens = request.max_completion_tokens or request.max_tokens or 256
    cfg = DecodeConfig(
        max_new_tokens=max_new_tokens,
        beta=request.beta if request.beta is not None else state.beta,
        alpha=request.alpha if request.alpha is not None else state.alpha,
        temperature=request.temperature or 0.0,
        top_k=request.top_k or 0,
        do_sample=bool(request.temperature and request.temperature > 0),
    )
    text = icd_generate(state.model, state.weak_model, state.tokenizer, original_prompt, weak_prompt, cfg)
    return text, original_prompt


@app.get("/health")
def health():
    return {"status": "ok", "model": state.model_name}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": state.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    content, prompt_text = generate_chat_completion(request)
    if request.stream:
        return StreamingResponse(stream_response(request, content), media_type="text/event-stream")
    return JSONResponse(make_completion_response(request, content, prompt_text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve ICD with an OpenAI-compatible API.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weak-model", default=None)
    parser.add_argument("--served-model-name", default="icd")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--weak-system-prompt", default=WEAK_SYSTEM_PROMPT)
    return parser


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    if args.beta <= 0:
        raise SystemExit("--beta must be greater than 0")
    if args.alpha < 0 or args.alpha > 1:
        raise SystemExit("--alpha must be between 0 and 1")

    model, weak_model, tokenizer = load_model_and_tokenizer(args)
    state.model = model
    state.weak_model = weak_model
    state.tokenizer = tokenizer
    state.model_name = args.served_model_name
    state.beta = args.beta
    state.alpha = args.alpha
    state.weak_system_prompt = args.weak_system_prompt
    state.no_chat_template = args.no_chat_template

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
