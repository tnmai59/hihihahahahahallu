from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str
    label: int

    @property
    def text(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


def load_truthfulqa_pairs(
    split: str = "validation",
    max_examples: int | None = None,
    seed: int | None = None,
    shuffle: bool = False,
) -> list[QAPair]:
    """Load TruthfulQA as answer-level binary QA pairs."""

    from datasets import load_dataset

    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    pairs: list[QAPair] = []
    for row in dataset:
        question = row["question"]
        for answer in row["correct_answers"]:
            pairs.append(QAPair(question=question, answer=answer, label=1))
        for answer in row["incorrect_answers"]:
            pairs.append(QAPair(question=question, answer=answer, label=0))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)
    if max_examples is not None:
        pairs = pairs[:max_examples]
    return pairs
