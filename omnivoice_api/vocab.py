"""What OmniVoice understands as a language and as a voice-design instruction.

OmniVoice's ``instruct`` is a closed vocabulary (gender, age, pitch, whisper, an English accent or
a Chinese dialect); upstream ``generate()`` raises ``ValueError`` for anything else, which fails
every line in the same call. Lines from Audiobook Studio carry free-form delivery notes ("tired,
flat"), so line instructions keep what OmniVoice knows and drop the rest, while a designed voice's
description is checked strictly when the voice is made.
"""

from __future__ import annotations

import re

from .vendor import voice_design
from .vendor.lang_map import LANG_IDS, LANG_NAME_TO_ID

EN_TO_ZH = voice_design._INSTRUCT_EN_TO_ZH
ZH_TO_EN = voice_design._INSTRUCT_ZH_TO_EN
CATEGORIES = voice_design._INSTRUCT_MUTUALLY_EXCLUSIVE
VALID_EN = voice_design._INSTRUCT_VALID_EN
ZH_RE = voice_design._ZH_RE

NONVERBAL_TAGS = [
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
]

_CATEGORY_OF = {item: index for index, category in enumerate(CATEGORIES) for item in category}
_ACCENTS = len(CATEGORIES) - 2
_DIALECTS = len(CATEGORIES) - 1


def resolve_language(language: str | None) -> str | None:
    """A language id OmniVoice knows, from an id, a BCP 47 tag or an English name; else None."""
    if not language or language.lower() == "none":
        return None
    for candidate in (language, language.split("-")[0].split("_")[0]):
        if candidate in LANG_IDS:
            return candidate
        if candidate.lower() in LANG_IDS:
            return candidate.lower()
    return LANG_NAME_TO_ID.get(language.lower())


def _split(instructions: str | None) -> list[str]:
    return [item.strip() for item in re.split(r"[,，;\n]", instructions or "") if item.strip()]


def unknown_instructions(instructions: str | None) -> list[str]:
    return [item for item in _split(instructions) if item.lower() not in _CATEGORY_OF]


def valid_instructions() -> list[str]:
    return sorted(VALID_EN)


def resolve_instruct(*sources: str | None, text: str = "") -> str | None:
    """Merge instruction strings into one OmniVoice instruct, later sources winning per category.

    Unknown items are dropped. The result is all-English or all-Chinese as OmniVoice requires:
    a dialect makes it Chinese, an accent English, otherwise it follows the text's script.
    """
    chosen: dict[int, str] = {}
    for source in sources:
        for raw in _split(source):
            item = raw.lower()
            category = _CATEGORY_OF.get(item)
            if category is not None:
                chosen[category] = item
    if _ACCENTS in chosen and _DIALECTS in chosen:
        del chosen[_ACCENTS if ZH_RE.search(text) else _DIALECTS]
    if not chosen:
        return None
    use_zh = _DIALECTS in chosen or (_ACCENTS not in chosen and bool(ZH_RE.search(text)))
    table = EN_TO_ZH if use_zh else ZH_TO_EN
    items = [table.get(chosen[category], chosen[category]) for category in sorted(chosen)]
    return ("，" if use_zh else ", ").join(items)


# Upstream's END_PUNCTUATION (omnivoice/utils/text.py): a transcript not ending in one of these
# gets a full stop, as ``create_voice_clone_prompt`` does before storing it.
END_PUNCTUATION = set(";:,.!?…)]}\"'“”‘’；：，。！？、）】") | {"……"}


def add_punctuation(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in END_PUNCTUATION:
        text += "。" if ZH_RE.search(text) else "."
    return text
