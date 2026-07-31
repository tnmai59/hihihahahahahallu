from __future__ import annotations

import csv
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ITIConfig, ITIDirections
from .hooks import ITIHook


@dataclass(frozen=True)
class MCExample:
    prompt: str
    choices: list[str]
    label: int | list[int]


@dataclass(frozen=True)
class QAExample:
    prompt: str
    answers: list[str]


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    total: int
    correct: int
    predictions: list[dict[str, Any]]


def load_builtin_mc_dataset(
    name: str,
    split: str = "validation",
    max_examples: int | None = None,
    subset: str | None = None,
) -> list[MCExample]:
    from datasets import load_dataset

    if name == "truthfulqa_mc1":
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
        examples = [
            MCExample(
                prompt=f"Q: {row['question']}\nA:",
                choices=list(row["mc1_targets"]["choices"]),
                label=int(row["mc1_targets"]["labels"].index(1)),
            )
            for row in dataset
        ]
    elif name == "truthfulqa_mc2":
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
        examples = [
            MCExample(
                prompt=f"Q: {row['question']}\nA:",
                choices=list(row["mc2_targets"]["choices"]),
                label=[idx for idx, label in enumerate(row["mc2_targets"]["labels"]) if int(label) == 1],
            )
            for row in dataset
        ]
    elif name == "mmlu":
        dataset = load_dataset("cais/mmlu", subset or "all", split=split)
        examples = [
            MCExample(
                prompt=format_mmlu_prompt(row),
                choices=[str(choice) for choice in row["choices"]],
                label=parse_mmlu_answer(row["answer"]),
            )
            for row in dataset
        ]
    elif name == "halueval":
        dataset = load_halueval_dataset(split=split, subset=subset)
        examples = [example for row in dataset for example in format_halueval_row(row)]
    elif name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split=split)
        examples = [
            MCExample(
                prompt=f"{row['ctx'].strip()}",
                choices=[str(choice) for choice in row["endings"]],
                label=int(row["label"]),
            )
            for row in dataset
        ]
    else:
        raise ValueError(f"Unknown built-in dataset: {name}")

    return examples[:max_examples] if max_examples is not None else examples


def load_builtin_qa_dataset(
    name: str,
    split: str = "validation",
    max_examples: int | None = None,
    subset: str | None = None,
) -> list[QAExample]:
    from datasets import load_dataset

    if name == "nq_open":
        dataset = load_dataset(subset or "nq_open", split=split)
        examples = [
            QAExample(
                prompt=f"Q: {row['question']}\nA:",
                answers=[str(answer) for answer in row["answer"]],
            )
            for row in dataset
        ]
    elif name == "natural_questions":
        dataset = load_dataset(subset or "google-research-datasets/natural_questions", split=split)
        examples = [format_natural_questions_row(row) for row in dataset]
        examples = [example for example in examples if example.answers]
    else:
        raise ValueError(f"Unknown built-in QA dataset: {name}")

    return examples[:max_examples] if max_examples is not None else examples


def load_custom_mc_dataset(path: str | Path, max_examples: int | None = None) -> list[MCExample]:
    """Load JSONL/CSV examples with columns: prompt, choices, label.

    `choices` may be a JSON list or a `|||` separated string.
    `label` is the zero-based index of the correct choice. For multiple correct
    answers, use a JSON list such as `[0, 2]`.
    """

    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("Custom eval files must be .jsonl or .csv.")

    examples: list[MCExample] = []
    for row in rows:
        raw_choices = row["choices"]
        if isinstance(raw_choices, str):
            choices = json.loads(raw_choices) if raw_choices.strip().startswith("[") else raw_choices.split("|||")
        else:
            choices = list(raw_choices)
        examples.append(
            MCExample(
                prompt=str(row["prompt"]),
                choices=[str(choice).strip() for choice in choices],
                label=parse_label(row["label"]),
            )
        )
    return examples[:max_examples] if max_examples is not None else examples


def load_custom_qa_dataset(path: str | Path, max_examples: int | None = None) -> list[QAExample]:
    """Load JSONL/CSV examples with columns: prompt, answers."""

    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("Custom eval files must be .jsonl or .csv.")

    examples: list[QAExample] = []
    for row in rows:
        raw_answers = row["answers"]
        if isinstance(raw_answers, str):
            answers = json.loads(raw_answers) if raw_answers.strip().startswith("[") else raw_answers.split("|||")
        else:
            answers = list(raw_answers)
        examples.append(QAExample(prompt=str(row["prompt"]), answers=[str(answer).strip() for answer in answers]))
    return examples[:max_examples] if max_examples is not None else examples


@torch.inference_mode()
def score_choices(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[MCExample],
    batch_size: int = 8,
    normalize_by_length: bool = True,
) -> list[dict[str, Any]]:
    model.eval()
    device = next(model.parameters()).device
    flat_items: list[tuple[int, int, str, int]] = []
    for example_idx, example in enumerate(examples):
        for choice_idx, choice in enumerate(example.choices):
            text = join_prompt_choice(example.prompt, choice)
            prefix_len = len(tokenizer(example.prompt, add_special_tokens=True)["input_ids"])
            flat_items.append((example_idx, choice_idx, text, prefix_len))

    scores_by_example = [[float("-inf")] * len(example.choices) for example in examples]
    loader = DataLoader(flat_items, batch_size=batch_size, shuffle=False, collate_fn=list)
    for batch in loader:
        texts = [item[2] for item in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        outputs = model(**encoded, use_cache=False)
        log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        for row_idx, (example_idx, choice_idx, _text, prefix_len) in enumerate(batch):
            seq_len = int(attention_mask[row_idx].sum().item())
            answer_start = max(prefix_len - 1, 0)
            answer_end = max(seq_len - 1, answer_start)
            answer_log_probs = token_log_probs[row_idx, answer_start:answer_end]
            score = answer_log_probs.sum().item()
            if normalize_by_length and answer_log_probs.numel() > 0:
                score /= answer_log_probs.numel()
            scores_by_example[example_idx][choice_idx] = score

    predictions: list[dict[str, Any]] = []
    for example, scores in zip(examples, scores_by_example):
        pred = int(torch.tensor(scores).argmax().item())
        predictions.append(
            {
                "prompt": example.prompt,
                "choices": example.choices,
                "label": example.label,
                "prediction": pred,
                "correct": pred in correct_labels(example.label),
                "scores": scores,
            }
        )
    return predictions


def evaluate_mc(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[MCExample],
    batch_size: int = 8,
    normalize_by_length: bool = True,
    directions: ITIDirections | None = None,
    alpha: float = 15.0,
) -> EvalResult:
    eval_config = ITIConfig(alpha=alpha, last_token_only=False)
    context = ITIHook(model, directions, eval_config) if directions is not None else nullcontext()
    with context:
        predictions = score_choices(
            model,
            tokenizer,
            examples,
            batch_size=batch_size,
            normalize_by_length=normalize_by_length,
        )

    correct = sum(1 for prediction in predictions if prediction["correct"])
    total = len(predictions)
    return EvalResult(
        accuracy=correct / total if total else 0.0,
        total=total,
        correct=correct,
        predictions=predictions,
    )


@torch.inference_mode()
def evaluate_qa_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[QAExample],
    max_new_tokens: int = 32,
    directions: ITIDirections | None = None,
    alpha: float = 15.0,
) -> EvalResult:
    model.eval()
    device = next(model.parameters()).device
    context = ITIHook(model, directions, ITIConfig(alpha=alpha)) if directions is not None else nullcontext()
    predictions: list[dict[str, Any]] = []

    with context:
        for example in examples:
            encoded = tokenizer(example.prompt, return_tensors="pt").to(device)
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_ids = output_ids[0, encoded["input_ids"].shape[1] :]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer = first_answer_line(generated)
            correct = any(normalize_answer(answer) == normalize_answer(gold) for gold in example.answers)
            predictions.append(
                {
                    "prompt": example.prompt,
                    "answers": example.answers,
                    "prediction": answer,
                    "raw_generation": generated,
                    "correct": correct,
                }
            )

    correct = sum(1 for prediction in predictions if prediction["correct"])
    total = len(predictions)
    return EvalResult(
        accuracy=correct / total if total else 0.0,
        total=total,
        correct=correct,
        predictions=predictions,
    )


def join_prompt_choice(prompt: str, choice: str) -> str:
    if not prompt:
        return choice
    if prompt[-1].isspace():
        return f"{prompt}{choice}"
    return f"{prompt} {choice}"


def parse_label(value: Any) -> int | list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return int(text)


def correct_labels(label: int | list[int]) -> set[int]:
    if isinstance(label, list):
        return {int(item) for item in label}
    return {int(label)}


def format_mmlu_prompt(row: dict[str, Any]) -> str:
    subject = str(row.get("subject", "")).replace("_", " ").strip()
    subject_line = f"The following are multiple choice questions about {subject}.\n\n" if subject else ""
    choices = [str(choice) for choice in row["choices"]]
    option_lines = "\n".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices))
    return f"{subject_line}Question: {row['question']}\n{option_lines}\nAnswer:"


def parse_mmlu_answer(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return ord(text.upper()[0]) - ord("A")


def load_halueval_dataset(split: str, subset: str | None):
    from datasets import load_dataset

    dataset_id = "pminervini/HaluEval"
    config = subset or "qa"
    if subset:
        try:
            return load_dataset(dataset_id, config, split=split)
        except Exception:
            return load_dataset(subset, split=split)
    try:
        return load_dataset(dataset_id, config, split=split)
    except Exception:
        return load_dataset(dataset_id, config, split="train")


def format_halueval_row(row: dict[str, Any]) -> list[MCExample]:
    question = first_present(row, ["question", "query", "prompt", "user_query", "instruction"])
    context = first_present(row, ["knowledge", "context", "document", "passage", "source"])
    dialogue = first_present(row, ["dialogue_history", "dialogue", "history"])
    right_answer = first_present(row, ["right_answer", "right_response", "right_summary", "reference", "ground_truth"])
    hallucinated_answer = first_present(
        row,
        ["hallucinated_answer", "hallucinated_response", "hallucinated_summary"],
    )
    answer = first_present(row, ["answer", "response", "model_answer", "output", "chatgpt_answer"])
    label_value = first_present(row, ["label", "hallucination", "is_hallucinated", "hallucinated"])

    examples: list[MCExample] = []
    if right_answer is not None and hallucinated_answer is not None and label_value is None and answer is None:
        examples.append(make_halueval_example(context, question, dialogue, right_answer, label=0))
        examples.append(make_halueval_example(context, question, dialogue, hallucinated_answer, label=1))
        return examples

    if answer is None:
        answer = hallucinated_answer if hallucinated_answer is not None else right_answer
    examples.append(make_halueval_example(context, question, dialogue, answer, label=parse_hallucination_label(label_value, row)))
    return examples


def make_halueval_example(
    context: Any,
    question: Any,
    dialogue: Any,
    answer: Any,
    label: int,
) -> MCExample:
    prompt_parts = []
    if context:
        prompt_parts.append(f"Context: {context}")
    if dialogue:
        prompt_parts.append(f"Dialogue: {dialogue}")
    if question:
        prompt_parts.append(f"Question: {question}")
    if answer:
        prompt_parts.append(f"Answer: {answer}")
    prompt_parts.append("Is the answer hallucinated? Reply yes or no.")

    return MCExample(
        prompt="\n".join(prompt_parts),
        choices=["no", "yes"],
        label=label,
    )


def parse_hallucination_label(value: Any, row: dict[str, Any]) -> int:
    if value is None:
        if first_present(row, ["hallucinated_answer"]):
            return 1
        raise ValueError(f"Could not infer HaluEval label from row keys: {sorted(row.keys())}")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "yes", "true", "hallucinated", "hallucination"}:
        return 1
    if text in {"0", "no", "false", "not_hallucinated", "factual", "grounded"}:
        return 0
    raise ValueError(f"Unknown HaluEval label: {value!r}")


def format_natural_questions_row(row: dict[str, Any]) -> QAExample:
    question = row.get("question")
    if isinstance(question, dict):
        question_text = str(question.get("text", ""))
    else:
        question_text = str(question)
    answers = extract_nq_answers(row)
    return QAExample(prompt=f"Q: {question_text}\nA:", answers=answers)


def extract_nq_answers(row: dict[str, Any]) -> list[str]:
    if "answer" in row:
        raw = row["answer"]
        return [str(item) for item in (raw if isinstance(raw, list) else [raw])]
    annotations = row.get("annotations") or {}
    short_answers = annotations.get("short_answers") if isinstance(annotations, dict) else None
    texts: list[str] = []
    if isinstance(short_answers, list):
        for answer in short_answers:
            if isinstance(answer, dict) and answer.get("text"):
                texts.append(str(answer["text"]))
            elif isinstance(answer, list):
                texts.extend(str(item.get("text", item)) for item in answer)
    if not texts and isinstance(annotations, dict) and annotations.get("yes_no_answer"):
        yes_no = annotations["yes_no_answer"]
        if isinstance(yes_no, list):
            texts.extend(str(item).lower() for item in yes_no if str(item).lower() not in {"none", "-1"})
        elif str(yes_no).lower() not in {"none", "-1"}:
            texts.append(str(yes_no).lower())
    return texts


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def first_answer_line(text: str) -> str:
    return text.strip().splitlines()[0].strip()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())
