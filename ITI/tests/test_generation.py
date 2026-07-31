from iti_paper.generation import apply_stop, messages_to_prompt


def test_messages_to_prompt_formats_chat_messages():
    prompt = messages_to_prompt(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert prompt == "System: Be concise.\nUser: Hello\nAssistant:"


def test_apply_stop_uses_earliest_stop_sequence():
    assert apply_stop("alpha beta gamma", ["gamma", " beta"]) == "alpha"
