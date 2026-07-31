from iti_paper.eval import (
    extract_nq_answers,
    format_halueval_row,
    format_mmlu_prompt,
    normalize_answer,
    parse_mmlu_answer,
)


def test_format_mmlu_prompt_and_answer_letter():
    row = {
        "subject": "abstract_algebra",
        "question": "What is the identity element?",
        "choices": ["0", "1", "x", "None"],
        "answer": "B",
    }

    prompt = format_mmlu_prompt(row)

    assert "abstract algebra" in prompt
    assert "A. 0" in prompt
    assert parse_mmlu_answer(row["answer"]) == 1


def test_format_halueval_row_binary_label():
    row = {
        "knowledge": "The Eiffel Tower is in Paris.",
        "question": "Where is the Eiffel Tower?",
        "answer": "Berlin",
        "hallucination": "yes",
    }

    example = format_halueval_row(row)[0]

    assert example.choices == ["no", "yes"]
    assert example.label == 1
    assert "Reply yes or no" in example.prompt


def test_format_halueval_row_expands_right_and_hallucinated_answers():
    row = {
        "knowledge": "The Eiffel Tower is in Paris.",
        "question": "Where is the Eiffel Tower?",
        "right_answer": "Paris",
        "hallucinated_answer": "Berlin",
    }

    examples = format_halueval_row(row)

    assert [example.label for example in examples] == [0, 1]
    assert "Paris" in examples[0].prompt
    assert "Berlin" in examples[1].prompt


def test_extract_nq_answers_from_open_format():
    row = {"question": "Who wrote Hamlet?", "answer": ["William Shakespeare", "Shakespeare"]}

    assert extract_nq_answers(row) == ["William Shakespeare", "Shakespeare"]


def test_normalize_answer_removes_articles_and_punctuation():
    assert normalize_answer("The Eiffel-Tower!") == "eiffel tower"
