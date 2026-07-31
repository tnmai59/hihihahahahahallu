from iti_paper.generation import apply_stop, merge_stop, messages_to_prompt


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

def test_merge_stop_adds_chat_transcript_stops():
    stops = merge_stop(None, ["\nUser:", "\nSystem:"])

    assert stops == ["\nUser:", "\nSystem:"]


def test_chat_transcript_stop_truncates_new_user_turn():
    text = "Hello! How can I assist?\nUser: another fake turn"

    assert apply_stop(text, merge_stop(None, ["\nUser:"])) == "Hello! How can I assist?"

