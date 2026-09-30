import pytest

from omnivoice_api.vocab import resolve_instruct, resolve_language, unknown_instructions


@pytest.mark.parametrize(
    "given, expected",
    [
        ("en", "en"),
        ("EN", "en"),
        ("en-GB", "en"),
        ("English", "en"),
        ("japanese", "ja"),
        (None, None),
        ("None", None),
        ("klingon", None),
    ],
)
def test_resolve_language(given, expected):
    assert resolve_language(given) == expected


def test_line_instructions_override_the_voice_per_category():
    assert resolve_instruct("male, low pitch, british accent", "High pitch, sad") == (
        "male, high pitch, british accent"
    )


def test_free_form_instructions_resolve_to_nothing():
    assert resolve_instruct("Tired, flat, under her breath.") is None
    assert unknown_instructions("whisper, tired") == ["tired"]


def test_chinese_text_gets_chinese_instructions():
    assert resolve_instruct("female, whisper", text="你好") == "女，耳语"
    assert resolve_instruct("女，四川话") == "女，四川话"
    assert resolve_instruct("male, american accent", text="你好") == "male, american accent"
