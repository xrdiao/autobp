from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def normalize_name(name):
    if name is None:
        return ""
    return str(name).strip().lower()


def slugify_name(name):
    text = normalize_name(name)
    text = text.replace(" - ", " ")
    text = text.replace("_", " ")
    text = text.replace("/", " ")
    return "-".join(text.split())
