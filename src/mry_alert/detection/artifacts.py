from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactTrimResult:
    text: str
    reason: str | None = None


def trim_repetitive_tail(text: str, minimum_repetitions: int = 5) -> ArtifactTrimResult:
    """Trim decoder repetition at the end while preserving normal short readbacks."""
    words = text.split()
    if len(words) < minimum_repetitions:
        return ArtifactTrimResult(text)
    cleaned = [re.sub(r"[^a-z0-9]", "", word.lower()) for word in words]
    for width in (1, 2, 3):
        for start in range(max(0, len(words) - 60), len(words)):
            tail = cleaned[start:]
            if len(tail) < width * minimum_repetitions or len(tail) % width:
                continue
            phrase = tail[:width]
            if phrase * (len(tail) // width) != tail:
                continue
            phrase_text = " ".join(phrase)
            kept = " ".join(words[:start]).strip()
            return ArtifactTrimResult(
                kept,
                f'Artifact trimming: removed {len(tail)} repeated "{phrase_text}" tokens',
            )
    return ArtifactTrimResult(text)
