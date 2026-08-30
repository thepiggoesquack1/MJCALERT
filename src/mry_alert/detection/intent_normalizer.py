from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntentNormalizationResult:
    text: str
    reasons: list[str] = field(default_factory=list)


def normalize_ground_intent(
    text: str, *, authorized_ground_source: bool
) -> IntentNormalizationResult:
    """Recover predictable Whisper errors only for the authorized KMRY ground path."""
    if not authorized_ground_source:
        return IntentNormalizationResult(text)
    normalized = text
    reasons: list[str] = []

    def replace(pattern: str, replacement: str, reason: str) -> None:
        nonlocal normalized
        updated, count = re.subn(pattern, replacement, normalized)
        if count:
            normalized = updated
            reasons.append(reason)

    ground_context = bool(
        re.search(
            r"\b(?:taxi|runway|foxtrot|alpha|echo|turn|continue|proceed|via|ground|jet)\b",
            normalized,
        )
    )
    if ground_context or re.search(r"\bsay\s+(?:barking|marking)\b", normalized):
        replace(
            r"\bsay\s+(?:barking|marking)\b",
            "say parking",
            "ATC parking-prompt pattern (confidence 0.92)",
        )
    replace(
        r"\bgoing\s+(?:on|for)\s+(?=(?:monterey\s+)?jet\b)",
        "going to ",
        "ground destination preposition recovery (confidence 0.91)",
    )
    replace(r"\bmonterey\s+jett\b", "monterey jet", "Jet Center name recovery")
    replace(r"\bjet\s+sender\b", "jet center", "Jet Center homophone recovery")
    if ground_context and re.search(r"\b(?:monterey\s+)?jet(?:\s+center)?\b", normalized):
        replace(r"\btech\s+to\b", "taxi to", "strong ground-routing context")
        replace(r"\btaxi\s+two\b", "taxi to", "strong destination context")
    callsign_context = bool(
        re.search(
            r"\b(?:november|citation|cessna|cherokee|bonanza|gulfstream|falcon|pilatus|cirrus)\b"
            r"(?:\s+[a-z0-9]+){1,7}",
            normalized,
        )
    )
    if callsign_context:
        replace(r"\btree\b", "three", "ATC callsign digit recovery")
        replace(r"\bfife\b", "five", "ATC callsign digit recovery")
        replace(r"\bfower\b", "four", "ATC callsign digit recovery")
    return IntentNormalizationResult(normalized, reasons)
