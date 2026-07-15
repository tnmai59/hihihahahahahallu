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


QA_INSTRUCTION = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question context and intention.
#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
#Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
#Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
#Your Judgement#: Yes
#Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
#Answer#: U.S Highway 70
#Your Judgement#: Yes

You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
#Question#: What genre do Superheaven and Oceansize belong to?
#Answer#: Superheaven and Oceansize belong to the rock genre.
#Your Judgement#: No
#Question#: What profession do Kobo Abe and Agatha Christie share?
#Answer#: Playwright.
#Your Judgement#: No

You are trying to determine if the answer can be correctly inferred from the knowledge.
#Question#: Which band has more members, Muse or The Raconteurs?
#Answer#: Muse has more members than The Raconteurs.
#Your Judgement#: Yes
#Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
#Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
#Your Judgement#: No

You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""


DIALOGUE_INSTRUCTION = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a highly similar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Steven Spielberg was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Batman Begins was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No
#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: United States of America was the director. He also directed insomnia and inception.
#Your Judgement#: Yes

You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""


SUMMARIZATION_INSTRUCTION = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
#Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.
#Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.
#Your Judgement#: Yes

You are trying to determine if there exists some non-factual and incorrect information in the summary.
#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
#Your Judgement#: No

You are trying to determine if there is a factual contradiction between the summary and the document.
#Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.
#Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.
#Your Judgement#: Yes

You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be "Yes" or "No"."""


SYSTEM_PROMPTS = {
    "qa": "You are a hallucination detector. You MUST determine if the provided answer contains hallucination or not for the question based on the world knowledge. The answer you provide MUST be \"Yes\" or \"No\".",
    "dialogue": "You are a response judge. You MUST determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\".",
    "summarization": "You are a summary judge. You MUST determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be \"Yes\" or \"No\".",
    "general": "You are a hallucination detector. You MUST determine if the provided response contains hallucination or not. The answer you provide MUST be \"Yes\" or \"No\".",
}


WEAK_SYSTEM_PROMPT = (
    "You are a careless hallucination detector. You make quick, superficial judgements and often "
    "miss subtle factual problems or overreact to harmless wording. Answer only Yes or No."
)


INSTRUCTIONS = {
    "qa": QA_INSTRUCTION,
    "dialogue": DIALOGUE_INSTRUCTION,
    "summarization": SUMMARIZATION_INSTRUCTION,
}


QA_ONE_SHOT_INSTRUCTION = """I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. The answer you give MUST be "Yes" or "No".

#Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
#Answer#: American Hairless Terrier
#Your Judgement#: No"""


DIALOGUE_ONE_SHOT_INSTRUCTION = """I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. The answer you give MUST be "Yes" or "No".

#Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
#Response#: Christopher Nolan was the director. He also directed insomnia and inception.
#Your Judgement#: No"""


SUMMARIZATION_ONE_SHOT_INSTRUCTION = """I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. The answer you give MUST be "Yes" or "No".

#Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". New South Wales Premier Mike Baird also announced plans for a permanent memorial in Martin Place.
#Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
#Your Judgement#: No"""


ONE_SHOT_INSTRUCTIONS = {
    "qa": QA_ONE_SHOT_INSTRUCTION,
    "dialogue": DIALOGUE_ONE_SHOT_INSTRUCTION,
    "summarization": SUMMARIZATION_ONE_SHOT_INSTRUCTION,
}


@dataclass
class Candidate:
    prompt_body: str
    text: str
    ground_truth: str
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
        choices=["minimal", "one-shot"],
        default="one-shot",
        help="minimal is zero-shot/simple; one-shot uses one HaluEval demonstration.",
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
    if prompt_style == "minimal":
        return 'Determine whether the provided text contains hallucinated or non-factual information. Answer only "Yes" or "No".'
    return ONE_SHOT_INSTRUCTIONS.get(
        task,
        'Determine whether the response contains hallucinated or non-factual information. Answer only "Yes" or "No".',
    )


def build_prompt_body(task: str, row: dict, candidate_text: str, use_knowledge: bool, prompt_style: str) -> str:
    instruction = task_instruction(task, prompt_style)
    parts = [instruction, ""]

    if task == "qa":
        if use_knowledge and row.get("knowledge"):
            parts.append("#Knowledge#: " + str(row["knowledge"]))
        parts.append("#Question#: " + str(row["question"]))
        parts.append("#Answer#: " + candidate_text)
    elif task == "dialogue":
        if use_knowledge and row.get("knowledge"):
            parts.append("#Knowledge#: " + str(row["knowledge"]))
        parts.append("#Dialogue History#: " + str(row["dialogue_history"]))
        parts.append("#Response#: " + candidate_text)
    elif task == "summarization":
        parts.append("#Document#: " + str(row["document"]))
        parts.append("#Summary#: " + candidate_text)
    else:
        query = row.get("user_query", row.get("query", row.get("instruction", "")))
        response = row.get("chatgpt_response", row.get("response", row.get("answer", candidate_text)))
        parts.append("#User Query#: " + str(query))
        parts.append("#Response#: " + str(response))

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
            prompt_body=build_prompt_body(task, row, text, use_knowledge, prompt_style),
            text=text,
            ground_truth=label,
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


def score_yes_no(model, weak_model, tokenizer, prompt_body: str, args) -> dict[str, float]:
    original_body = truncate_prompt(tokenizer, prompt_body, args.max_input_tokens)
    system_prompt = args.system_prompt or SYSTEM_PROMPTS[args.task]
    original_prompt = format_prompt(tokenizer, system_prompt, original_body, args.no_chat_template)
    weak_prompt = format_prompt(tokenizer, args.weak_system_prompt, original_body, args.no_chat_template)

    if args.mode == "original":
        return {
            "Yes": score_label_original(model, tokenizer, original_prompt, "Yes"),
            "No": score_label_original(model, tokenizer, original_prompt, "No"),
        }

    return {
        "Yes": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "Yes", args.beta, args.alpha),
        "No": score_label_icd(model, weak_model, tokenizer, original_prompt, weak_prompt, "No", args.beta, args.alpha),
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

    prediction = max(scores, key=scores.get)
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
            )
        )

    model, weak_model, tokenizer = load_model_and_tokenizer(args)
    calibration_scores = None
    if args.label_prior_calibration and args.decision_mode == "likelihood":
        calibration_scores = score_yes_no(model, weak_model, tokenizer, calibration_prompt_body(args.task, args.prompt_style), args)
    output_file = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    tp = fp = tn = fn = invalid = 0
    try:
        for candidate in tqdm(candidates, desc=f"HaluEval {args.task}"):
            prediction, scores = classify_candidate(model, weak_model, tokenizer, candidate, args, calibration_scores)
            correct = prediction == candidate.ground_truth
            if prediction == "failed":
                invalid += 1
            elif candidate.ground_truth == "Yes" and prediction == "Yes":
                tp += 1
            elif candidate.ground_truth == "No" and prediction == "Yes":
                fp += 1
            elif candidate.ground_truth == "No" and prediction == "No":
                tn += 1
            else:
                fn += 1

            if output_file:
                output_file.write(
                    json.dumps(
                        {
                            "index": candidate.source_index,
                            "task": args.task,
                            "candidate_type": candidate.candidate_type,
                            "text": candidate.text,
                            "ground_truth": candidate.ground_truth,
                            "prediction": prediction,
                            "correct": correct,
                            "scores_or_generation": scores,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        print(json.dumps(metrics_from_counts(tp, fp, tn, fn, invalid), indent=2))
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
