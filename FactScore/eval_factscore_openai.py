#!/usr/bin/env python3
"""FactScore-style evaluation for any OpenAI-compatible instruct model.

Example:
    python eval_factscore_openai.py \
      --model Qwen3-32B \
      --server-host http://0.0.0.0:8001 \
      --topics-file unlabeled/prompt_entities.txt \
      --output-path outputs/factscore_qwen3.jsonl

The evaluated model only needs to expose /v1/chat/completions, e.g. through
vLLM. The script generates biographies, atomizes them, judges the atomic facts,
and prints final FactScore-style metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from multiprocessing import Pool
from typing import Any

from tqdm.auto import tqdm


# ============================
# Configuration
# ============================

DEFAULT_MODEL = "Qwen3-32B"
SERVER_HOST = "http://0.0.0.0:8001"
API_KEY = "EMPTY"
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512
ATOMIZER_MAX_TOKENS = 768
JUDGE_MAX_TOKENS = 256
N_PROCESSES = 8
THINK = True

TOPICS_FILE = "unlabeled/prompt_entities.txt"
DATA_PATH = None
OUTPUT_PATH = "factscore_predictions.jsonl"
SUMMARY_PATH = None


GENERATION_SYSTEM_PROMPT = (
    "You are a helpful, careful assistant. Answer factual questions accurately. "
    "If you do not know, say so instead of making up information."
)

ATOMIZER_SYSTEM_PROMPT = (
    "You extract atomic factual claims from biographies. Return only valid JSON."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict factuality judge. Judge claims using your world knowledge. "
    "Return only valid JSON."
)

NON_RESPONSE_PATTERNS = [
    r"\bi (do not|don't|cannot|can't) (know|provide|find)\b",
    r"\bnot enough (information|context)\b",
    r"\bwithout (more|additional|specific) (information|context|details)\b",
    r"\bplease provide (more|additional|specific) (information|context|details)\b",
    r"\bi'?m sorry\b",
]


# ============================
# LLM Client
# ============================

def api_base(server_host: str) -> str:
    server_host = server_host.rstrip("/")
    return server_host if server_host.endswith("/v1") else f"{server_host}/v1"


def get_completion(
    user_prompt: str,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    server_host: str = SERVER_HOST,
    api_key: str = API_KEY,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    think: bool = THINK,
    max_retries: int = 4,
    retry_sleep: float = 5.0,
) -> str | None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base(server_host))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if think:
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": "/no_think " + user_prompt})

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=0,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[ERROR] LLM request failed on attempt {attempt + 1}: {exc}")
            if attempt >= max_retries:
                return None
            time.sleep(retry_sleep * (attempt + 1))
    return None


# ============================
# Data Loading
# ============================

def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def topic_from_prompt(prompt: str) -> str:
    match = re.search(r"bio of (.+?)[.?]\s*$", prompt, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def load_examples(args) -> list[dict[str, Any]]:
    examples = []
    if args.data_path:
        for idx, row in enumerate(load_jsonl(args.data_path)):
            topic = str(row.get("topic") or row.get("entity") or "").strip()
            prompt = str(row.get("input") or "").strip()
            if not topic and prompt:
                topic = topic_from_prompt(prompt)
            if not topic:
                raise ValueError(f"Row {idx} has no topic/entity field.")
            examples.append(
                {
                    "index": idx,
                    "topic": topic,
                    "input": prompt or args.prompt_template.format(topic=topic),
                    "existing_output": row.get("output"),
                    "raw": row,
                }
            )
    else:
        with open(args.topics_file, "r", encoding="utf-8") as file:
            topics = [line.strip() for line in file if line.strip()]
        for idx, topic in enumerate(topics):
            examples.append(
                {
                    "index": idx,
                    "topic": topic,
                    "input": args.prompt_template.format(topic=topic),
                    "existing_output": None,
                    "raw": {},
                }
            )

    end = None if args.max_examples is None else args.start + args.max_examples
    return examples[args.start : min(end or len(examples), len(examples))]


# ============================
# FactScore Task
# ============================

def extract_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = re.findall(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from model output: {text[:500]}")


def normalize_facts(parsed: Any) -> list[str]:
    if isinstance(parsed, dict):
        parsed = parsed.get("facts", parsed.get("atomic_facts", []))

    facts = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                fact = item.strip()
            elif isinstance(item, dict):
                fact = str(item.get("fact") or item.get("text") or "").strip()
            else:
                fact = ""
            if fact and fact not in facts:
                facts.append(fact)
    return facts


def is_non_response(text: str) -> bool:
    stripped = text.strip()
    if len(stripped.split()) < 8:
        return True
    lowered = stripped.lower()
    return any(re.search(pattern, lowered) for pattern in NON_RESPONSE_PATTERNS)


def atomize_bio(topic: str, bio: str, args) -> tuple[list[str], str | None]:
    if is_non_response(bio):
        return [], None

    prompt = (
        "Break the biography into decontextualized atomic facts about the person.\n"
        "Rules:\n"
        "- Each fact must be a short standalone sentence.\n"
        "- Replace pronouns with the person's name when needed.\n"
        "- Do not include opinions, vague praise, or duplicate facts.\n"
        "- Return JSON exactly as {\"facts\": [\"...\"]}.\n\n"
        f"Person: {topic}\n"
        f"Biography:\n{bio}"
    )
    text = get_completion(
        prompt,
        system_prompt=ATOMIZER_SYSTEM_PROMPT,
        model=args.judge_model,
        server_host=args.judge_server_host,
        api_key=args.judge_api_key,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.atomizer_max_tokens,
        think=args.judge_think,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    if text is None:
        return [], "atomizer_error: request failed"

    try:
        return normalize_facts(extract_json(text)), None
    except Exception as exc:
        return [], f"atomizer_error: {exc}"


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"s", "supported", "support", "true", "yes"}:
        return "S"
    if text in {"ns", "unsupported", "not supported", "false", "no"}:
        return "NS"
    if text in {"ir", "irrelevant", "not a factual claim", "unclear"}:
        return "IR"
    return "NS"


def judge_fact(topic: str, fact: str, args) -> dict[str, str]:
    prompt = (
        "Judge whether the atomic fact is true and directly about the person.\n"
        "Use labels:\n"
        "- S: supported/true\n"
        "- NS: not supported/false or unverifiable\n"
        "- IR: irrelevant or not a factual claim\n"
        "Return JSON exactly as {\"label\": \"S|NS|IR\", \"reason\": \"short reason\"}.\n\n"
        f"Person: {topic}\n"
        f"Atomic fact: {fact}"
    )
    text = get_completion(
        prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=args.judge_model,
        server_host=args.judge_server_host,
        api_key=args.judge_api_key,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
        think=args.judge_think,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    if text is None:
        return {"label": "NS", "reason": "judge_error: request failed"}

    try:
        parsed = extract_json(text)
        if not isinstance(parsed, dict):
            raise ValueError("judge returned non-object JSON")
        return {
            "label": normalize_label(parsed.get("label")),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except Exception as exc:
        return {"label": "NS", "reason": f"judge_error: {exc}"}


def length_penalty(num_facts: int, gamma: float) -> float:
    if gamma <= 0 or num_facts >= gamma:
        return 1.0
    if num_facts <= 0:
        return 0.0
    return math.exp(1.0 - gamma / num_facts)


def score_labels(labels: list[str], gamma: float) -> dict[str, Any]:
    supported = sum(1 for label in labels if label == "S")
    unsupported = sum(1 for label in labels if label == "NS")
    irrelevant = sum(1 for label in labels if label == "IR")
    scored_facts = supported + unsupported
    init_score = supported / scored_facts if scored_facts else 0.0
    penalty = length_penalty(scored_facts, gamma)
    return {
        "score": init_score * penalty,
        "init_score": init_score,
        "length_penalty": penalty,
        "num_supported": supported,
        "num_unsupported": unsupported,
        "num_irrelevant": irrelevant,
        "num_facts": scored_facts,
    }


def factscore_task(task):
    example, args = task

    if args.use_existing_output and example.get("existing_output"):
        bio = str(example["existing_output"]).strip()
    else:
        bio = get_completion(
            example["input"],
            system_prompt=args.system_prompt,
            model=args.model,
            server_host=args.server_host,
            api_key=args.api_key,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            think=args.think,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        if bio is None:
            bio = ""

    facts, atomizer_error = atomize_bio(example["topic"], bio, args)
    judgements = [judge_fact(example["topic"], fact, args) for fact in facts]
    labels = [judgement["label"] for judgement in judgements]
    scores = score_labels(labels, args.gamma)

    return {
        "index": example["index"],
        "topic": example["topic"],
        "input": example["input"],
        "output": bio,
        "atomizer_error": atomizer_error,
        "atomic_facts": [
            {"text": fact, "label": judgement["label"], "reason": judgement["reason"]}
            for fact, judgement in zip(facts, judgements)
        ],
        **scores,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_factscore(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row["num_facts"] > 0]
    total_supported = sum(row["num_supported"] for row in rows)
    total_unsupported = sum(row["num_unsupported"] for row in rows)
    total_facts = total_supported + total_unsupported

    return {
        "factscore": round(100 * mean([row["score"] for row in rows]), 2),
        "factscore_no_length_penalty": round(100 * mean([row["init_score"] for row in rows]), 2),
        "respond_ratio": round(100 * len(valid_rows) / len(rows), 2) if rows else 0.0,
        "num_facts_per_valid_response": round(mean([row["num_facts"] for row in valid_rows]), 2),
        "micro_factscore": round(100 * total_supported / total_facts, 2) if total_facts else 0.0,
        "num_supported": total_supported,
        "num_unsupported": total_unsupported,
        "num_irrelevant": sum(row["num_irrelevant"] for row in rows),
        "num_facts": total_facts,
        "total": len(rows),
    }


# ============================
# Main
# ============================

def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate FactScore through an OpenAI-compatible API.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server-host", default=SERVER_HOST)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", API_KEY))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-server-host", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--topics-file", default=TOPICS_FILE)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-path", default=OUTPUT_PATH)
    parser.add_argument("--summary-path", default=SUMMARY_PATH)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-processes", type=int, default=N_PROCESSES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--atomizer-max-tokens", type=int, default=ATOMIZER_MAX_TOKENS)
    parser.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--system-prompt", default=GENERATION_SYSTEM_PROMPT)
    parser.add_argument("--prompt-template", default="Tell me a bio of {topic}.")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--use-existing-output", action="store_true")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=THINK)
    parser.add_argument("--judge-think", action=argparse.BooleanOptionalAction, default=THINK)
    return parser


def main():
    args = build_parser().parse_args()
    args.judge_model = args.judge_model or args.model
    args.judge_server_host = args.judge_server_host or args.server_host
    args.judge_api_key = args.judge_api_key or args.api_key

    print("Loading FactScore topics...")
    examples = load_examples(args)
    print(f"Loaded {len(examples)} examples.")
    print(f"Generation model: {args.model} at {api_base(args.server_host)}")
    print(f"Judge model: {args.judge_model} at {api_base(args.judge_server_host)}")

    tasks = [(example, args) for example in examples]
    if args.n_processes == 1:
        rows = [factscore_task(task) for task in tqdm(tasks)]
    else:
        print(f"Running FactScore evaluation with {args.n_processes} processes...")
        with Pool(args.n_processes) as pool:
            rows = list(tqdm(pool.imap(factscore_task, tasks), total=len(tasks)))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = evaluate_factscore(rows)
    summary_path = args.summary_path or args.output_path + ".summary.json"
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\n===== FINAL RESULTS =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Finish evaluation!!")


if __name__ == "__main__":
    main()
