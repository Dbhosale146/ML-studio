"""Versioned prompt templates (*.vN.md). Placeholders use {{name}}; render() never evaluates anything."""

from functools import cache
from pathlib import Path

DIR = Path(__file__).parent


@cache
def _load(name: str) -> str:
    return (DIR / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **values: object) -> str:
    text = _load(name)
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text
