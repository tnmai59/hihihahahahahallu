#!/usr/bin/env python3
"""Evaluate hallucination detection on HaluEval JSON/JSONL files.

The model is used as a binary judge. For each candidate response, we score the
continuations "Yes" and "No" after the HaluEval prompt:

  Yes = the candidate contains hallucination
  No  = the candidate does not contain hallucination

In ICD mode, label tokens are scored with:
  log_softmax(beta * logprob(original) - logprob(weak))
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

from icd_generate import DecodeConfig, format_prompt, icd_generate, load_model_and_tokenizer


SYSTEM_PROMPTS = {
    "qa": "You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provide MUST be \"Yes\" or \"No\".",
    "dialogue": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\".",
    "summarization": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\".",
    "general": "You are a hallucination detector. You MUST determine if the provided response contains hallucination or not. The answer you provide MUST be \"Yes\" or \"No\".",
}


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
class Candidate:
    prompt_body: str
    text: str
    ground_truth: str
    choices: list[str]
    source_index: int
    candidate_type: str
    raw: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate HaluEval hallucination detection.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weak-model", default=None)
    parser.add_argument("--task", choices=["qa", "dialogue", "summarization", "general"], default="qa")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--mode", choices=["original", "icd"], default="icd")
    parser.add_argument(
        "--decision-mode",
        choices=["likelihood", "generate"],
        default="likelihood",
        help="likelihood scores Yes/No labels directly; generate decodes a short Yes/No answer.",
    )
    parser.add_argument("--candidate-mode", choices=["both", "random", "right", "hallucinated"], default="both")
    parser.add_argument(
        "--prompt-format",
        choices=["judge", "mc"],
        default="mc",
        help="mc converts each sample to a Yes/No multiple-choice question.",
    )
    parser.add_argument(
        "--audit-modes",
        action="store_true",
        help="Report original, weak, and ICD metrics on the same converted samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--use-knowledge", action="store_true")
    parser.add_argument(
        "--prompt-style",
        choices=["none", "minimal"],
        default="none",
        help="none uses no instruction; minimal is a short zero-shot instruction.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--weak-system-prompt", default=WEAK_SYSTEM_PROMPT)
    parser.add_argument(
        "--label-prior-calibration",
        action="store_true",
        help="Subtract Yes/No scores measured on an empty judgement prompt to reduce label prior bias.",
    )
    return parser


def load_json_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        text = file.read().strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of objects.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_label(label) -> str:
    text = str(label).strip().lower().strip(".")
    if text in {"yes", "y", "true", "1", "hallucinated", "hallucination"}:
        return "Yes"
    if text in {"no", "n", "false", "0", "factual", "not hallucinated", "non-hallucinated"}:
        return "No"
    raise ValueError(f"Cannot normalize label: {label!r}")


def task_instruction(task: str, prompt_style: str) -> str:
    if prompt_style == "none":
        return ""
    if prompt_style == "minimal":
        return 'Determine whether the provided text contains hallucinated or non-factual information. Answer only "Yes" or "No".'
    return ""


def build_prompt_body(
    task: str,
    row: dict,
    candidate_text: str,
    use_knowledge: bool,
    prompt_style: str,
    prompt_format: str,
) -> str:
    instruction = task_instruction(task, prompt_style)
    parts = [instruction, ""] if instruction else []

    if task == "qa":
        if use_knowledge and row.get("knowledge"):
            parts.append("#Knowledge#: " + str(row["knowledge"]))
        parts.append("#Question#: " + str(row["question"]))
        parts.append("#Candidate Answer#: " + candidate_text)
    elif task == "dialogue":
        if use_knowledge and row.get("knowledge"):
            parts.append("#Knowledge#: " + str(row["knowledge"]))
        parts.append("#Dialogue History#: " + str(row["dialogue_history"]))
        parts.append("#Candidate Response#: " + candidate_text)
    elif task == "summarization":
        parts.append("#Document#: " + str(row["document"]))
        parts.append("#Candidate Summary#: " + candidate_text)
    else:
        query = row.get("user_query", row.get("query", row.get("instruction", "")))
        response = row.get("chatgpt_response", row.get("response", row.get("answer", candidate_text)))
        parts.append("#User Query#: " + str(query))
        parts.append("#Response#: " + str(response))

    if prompt_format == "mc":
        parts.append("#Multiple Choice Question#: Does the provided text contain hallucinated or non-factual information?")
        parts.append("#Choices#: Yes / No")
        parts.append("#Correct Choice#: ")
    else:
        parts.append("#Your Judgement#: ")
    return "\n".join(parts)


def row_candidates(
    task: str,
    row: dict,
    index: int,
    candidate_mode: str,
    rng: random.Random,
    use_knowledge: bool,
    prompt_style: str,
    prompt_format: str,
) -> list[Candidate]:
    candidates: list[tuple[str, str, str]] = []

    if task == "qa":
        candidates = [
            ("right", str(row["right_answer"]), "No"),
            ("hallucinated", str(row["hallucinated_answer"]), "Yes"),
        ]
    elif task == "dialogue":
        candidates = [
            ("right", str(row["right_response"]), "No"),
            ("hallucinated", str(row["hallucinated_response"]), "Yes"),
        ]
    elif task == "summarization":
        candidates = [
            ("right", str(row["right_summary"]), "No"),
            ("hallucinated", str(row["hallucinated_summary"]), "Yes"),
        ]
    else:
        response = str(row.get("chatgpt_response", row.get("response", row.get("answer", ""))))
        label = row.get("hallucination_label", row.get("hallucination", row.get("label")))
        candidates = [("annotated", response, normalize_label(label))]

    if candidate_mode == "random" and len(candidates) > 1:
        candidates = [rng.choice(candidates)]
    elif candidate_mode == "right":
        candidates = [candidate for candidate in candidates if candidate[0] in {"right", "annotated"}]
    elif candidate_mode == "hallucinated":
        candidates = [candidate for candidate in candidates if candidate[0] in {"hallucinated", "annotated"}]

    return [
        Candidate(
            prompt_body=build_prompt_body(task, row, text, use_knowledge, prompt_style, prompt_format),
            text=text,
            ground_truth=label,
            choices=["Yes", "No"],
            source_index=index,
            candidate_type=kind,
            raw=row,
        )
        for kind, text, label in candidates
    ]


def encode(tokenizer, text: str, device):
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def truncate_prompt(tokenizer, prompt: str, max_input_tokens: int) -> str:
    if max_input_tokens <= 0:
        return prompt
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(ids) <= max_input_tokens:
        return prompt
    ids = ids[-max_input_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def label_ids(tokenizer, label: str, device):
    return tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def score_label_original(model, tokenizer, prompt: str, label: str) -> float:
    import torch

    device = model.get_input_embeddings().weight.device
    prompt_ids = encode(tokenizer, prompt, device)
    continuation_ids = label_ids(tokenizer, label, device)
    input_ids = torch.cat([prompt_ids, continuation_ids], dim=-1)

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        logprobs = torch.log_softmax(logits, dim=-1)

    start = prompt_ids.shape[-1] - 1
    total = 0.0
    for i, token_id in enumerate(continuation_ids[0]):
        total += float(logprobs[0, start + i, int(token_id)])
    return total


def score_label_with_model(model, tokenizer, prompt: str, label: str) -> float:
    return score_label_original(model, tokenizer, prompt, label)


def score_label_icd(model, weak_model, tokenizer, original_prompt: str, weak_prompt: str, label: str, beta: float, alpha: float) -> float:
    import torch

    original_device = model.get_input_embeddings().weight.device
    weak_device = weak_model.get_input_embeddings().weight.device
    original_prompt_ids = encode(tokenizer, original_prompt, original_device)
    weak_prompt_ids = encode(tokenizer, weak_prompt, weak_device)
    continuation_ids = label_ids(tokenizer, label, original_device)
    weak_continuation_ids = continuation_ids.to(weak_device)

    original_input_ids = torch.cat([original_prompt_ids, continuation_ids], dim=-1)
    weak_input_ids = torch.cat([weak_prompt_ids, weak_continuation_ids], dim=-1)

    with torch.inference_mode():
        original_logits = model(input_ids=original_input_ids).logits
        weak_logits = weak_model(input_ids=weak_input_ids).logits.to(original_logits.device)
        original_logprobs = torch.log_softmax(original_logits, dim=-1)
        weak_logprobs = torch.log_softmax(weak_logits, dim=-1)

        original_start = original_prompt_ids.shape[-1] - 1
        weak_start = weak_prompt_ids.shape[-1] - 1
        total = 0.0
        for i, token_id in enumerate(continuation_ids[0]):
            original_step = original_start + i
            weak_step = weak_start + i
            diff_logits = beta * original_logprobs[0, original_step, :] - weak_logprobs[0, weak_step, :]
            if alpha > 0:
                original_probs = torch.softmax(original_logits[0, original_step, :], dim=-1)
                threshold = alpha * torch.max(original_probs)
                diff_logits = diff_logits.masked_fill(original_probs < threshold, -float("inf"))
            diff_logprobs = torch.log_softmax(diff_logits, dim=-1)
            total += float(diff_logprobs[int(token_id)])
    return total


def make_prompts(tokenizer, prompt_body: str, args) -> tuple[str, str]:
    original_body = truncate_prompt(tokenizer, prompt_body, args.max_input_tokens)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS[args.task]
    original_prompt = format_prompt(tokenizer, system_prompt, original_body, args.no_chat_template)
    weak_prompt = format_prompt(tokenizer, args.weak_system_prompt, original_body, args.no_chat_template)
    return original_prompt, weak_prompt


def score_yes_no(model, weak_model, tokenizer, prompt_body: str, args) -> dict[str, float]:
    original_prompt, weak_prompt = make_prompts(tokenizer, prompt_body, args)

    if args.mode == "original":
        return {
            "Yes": score_label_original(model, tokenizer, original_prompt, "Yes"),
            "No": score_label_original(model, tokenizer, original_prompt, "No"),
        }

    return {
        "Yes": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "Yes", args.beta, args.alpha),
        "No": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "No", args.beta, args.alpha),
    }


def score_all_modes(model, weak_model, tokenizer, prompt_body: str, args) -> dict[str, dict[str, float]]:
    original_prompt, weak_prompt = make_prompts(tokenizer, prompt_body, args)
    return {
        "original": {
            "Yes": score_label_original(model, tokenizer, original_prompt, "Yes"),
            "No": score_label_original(model, tokenizer, original_prompt, "No"),
        },
        "weak": {
            "Yes": score_label_with_model(weak_model, tokenizer, weak_prompt, "Yes"),
            "No": score_label_with_model(weak_model, tokenizer, weak_prompt, "No"),
        },
        "icd": {
            "Yes": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "Yes", args.beta, args.alpha),
            "No": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "No", args.beta, args.alpha),
        },
    }


def parse_yes_no(text: str) -> str | None:
    cleaned = text.strip().replace(".", " ").replace(",", " ")
    words = [word.strip().lower() for word in cleaned.split()]
    has_yes = "yes" in words
    has_no = "no" in words
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return None


def generate_judgement(model, weak_model, tokenizer, prompt_body: str, args) -> tuple[str | None, dict[str, float | str]]:
    original_body = truncate_prompt(tokenizer, prompt_body, args.max_input_tokens)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS[args.task]
    original_prompt = format_prompt(tokenizer, system_prompt, original_body, args.no_chat_template)
    weak_prompt = format_prompt(tokenizer, args.weak_system_prompt, original_body, args.no_chat_template)

    if args.mode == "original":
        import torch

        device = model.get_input_embeddings().weight.device
        input_ids = encode(tokenizer, original_prompt, device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
    else:
        cfg = DecodeConfig(
            max_new_tokens=args.max_new_tokens,
            beta=args.beta,
            alpha=args.alpha,
            temperature=0.0,
            top_k=0,
            do_sample=False,
        )
        text = icd_generate(model, weak_model, tokenizer, original_prompt, weak_prompt, cfg)

    return parse_yes_no(text), {"generated": text}


def calibration_prompt_body(task: str, prompt_style: str) -> str:
    return task_instruction(task, prompt_style) + "\n\n#Your Judgement#: "


def classify_candidate(
    model,
    weak_model,
    tokenizer,
    candidate: Candidate,
    args,
    calibration_scores: dict[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    if args.decision_mode == "generate":
        prediction, generation_info = generate_judgement(model, weak_model, tokenizer, candidate.prompt_body, args)
        if prediction is None:
            return "failed", generation_info
        return prediction, generation_info

    scores = score_yes_no(model, weak_model, tokenizer, candidate.prompt_body, args)
    if calibration_scores:
        scores = {label: score - calibration_scores[label] for label, score in scores.items()}

    prediction = max(candidate.choices, key=lambda choice: scores[choice])
    return prediction, scores


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int, invalid: int) -> dict:
    total = tp + fp + tn + fn + invalid
    valid_total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    valid_accuracy = (tp + tn) / valid_total if valid_total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
        "precision_yes": precision,
        "recall_yes": recall,
        "f1_yes": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "invalid": invalid,
        "total": total,
        "valid_total": valid_total,
    }


def update_counts(counts: dict[str, int], ground_truth: str, prediction: str) -> None:
    if prediction == "failed":
        counts["invalid"] += 1
    elif ground_truth == "Yes" and prediction == "Yes":
        counts["tp"] += 1
    elif ground_truth == "No" and prediction == "Yes":
        counts["fp"] += 1
    elif ground_truth == "No" and prediction == "No":
        counts["tn"] += 1
    else:
        counts["fn"] += 1


def main() -> None:
    from tqdm import tqdm

    args = build_parser().parse_args()
    if args.beta <= 0:
        raise SystemExit("--beta must be greater than 0")
    if args.alpha < 0 or args.alpha > 1:
        raise SystemExit("--alpha must be between 0 and 1")

    rng = random.Random(args.seed)
    rows = load_json_records(args.dataset_jsonl)
    end = None if args.max_examples is None else args.start + args.max_examples
    selected_rows = rows[args.start:min(end or len(rows), len(rows))]

    candidates: list[Candidate] = []
    for offset, row in enumerate(selected_rows):
        candidates.extend(
            row_candidates(
                args.task,
                row,
                args.start + offset,
                args.candidate_mode,
                rng,
                args.use_knowledge,
                args.prompt_style,
                args.prompt_format,
            )
        )

    model, weak_model, tokenizer = load_model_and_tokenizer(args)
    calibration_scores = None
    if args.label_prior_calibration and args.decision_mode == "likelihood":
        calibration_scores = score_yes_no(model, weak_model, tokenizer, calibration_prompt_body(args.task, args.prompt_style), args)
    output_file = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    tp = fp = tn = fn = invalid = 0
    audit_counts = {
        "original": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "invalid": 0},
        "weak": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "invalid": 0},
        "icd": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "invalid": 0},
    }
    try:
        for candidate in tqdm(candidates, desc=f"HaluEval {args.task}"):
            audit_result = None
            if args.audit_modes and args.decision_mode == "likelihood":
                all_scores = score_all_modes(model, weak_model, tokenizer, candidate.prompt_body, args)
                audit_result = {}
                for mode_name, mode_scores in all_scores.items():
                    mode_prediction = max(candidate.choices, key=lambda choice: mode_scores[choice])
                    update_counts(audit_counts[mode_name], candidate.ground_truth, mode_prediction)
                    audit_result[mode_name] = {
                        "prediction": mode_prediction,
                        "scores": mode_scores,
                    }
                prediction = audit_result[args.mode]["prediction"] if args.mode in audit_result else audit_result["icd"]["prediction"]
                scores = audit_result[args.mode]["scores"] if args.mode in audit_result else audit_result["icd"]["scores"]
            else:
                prediction, scores = classify_candidate(model, weak_model, tokenizer, candidate, args, calibration_scores)
            correct = prediction == candidate.ground_truth
            main_counts = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "invalid": invalid}
            update_counts(main_counts, candidate.ground_truth, prediction)
            tp = main_counts["tp"]
            fp = main_counts["fp"]
            tn = main_counts["tn"]
            fn = main_counts["fn"]
            invalid = main_counts["invalid"]

            if output_file:
                row = {
                    "index": candidate.source_index,
                    "task": args.task,
                    "candidate_type": candidate.candidate_type,
                    "text": candidate.text,
                    "choices": candidate.choices,
                    "ground_truth": candidate.ground_truth,
                    "prediction": prediction,
                    "correct": correct,
                    "scores_or_generation": scores,
                }
                if audit_result is not None:
                    row["audit"] = audit_result
                output_file.write(json.dumps(row, ensure_ascii=True) + "\n")

        result = metrics_from_counts(tp, fp, tn, fn, invalid)
        if args.audit_modes and args.decision_mode == "likelihood":
            result["audit_modes"] = {
                mode_name: metrics_from_counts(
                    counts["tp"],
                    counts["fp"],
                    counts["tn"],
                    counts["fn"],
                    counts["invalid"],
                )
                for mode_name, counts in audit_counts.items()
            }
        print(json.dumps(result, indent=2))
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
