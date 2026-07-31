#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve ITI behind an OpenAI-compatible API.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--directions", help="Optional ITI directions file from scripts/train_iti.py.")
    parser.add_argument("--model-id", help="Model id exposed by /v1/models.")
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--api-key", help="Optional bearer token expected from clients.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["ITI_MODEL"] = args.model
    os.environ["ITI_ALPHA"] = str(args.alpha)
    os.environ["ITI_DTYPE"] = args.dtype
    os.environ["ITI_DEVICE_MAP"] = args.device_map
    if args.directions:
        os.environ["ITI_DIRECTIONS"] = args.directions
    if args.model_id:
        os.environ["ITI_MODEL_ID"] = args.model_id
    if args.api_key:
        os.environ["ITI_API_KEY"] = args.api_key

    uvicorn.run("iti_paper.openai_api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
