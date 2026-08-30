import re

REPLACEMENTS = {
    r"\bmonterrey\b": "monterey",
    r"\bjett\s+center\b": "jet center",
    r"\bjetcenter\b": "jet center",
    r"\bx[\s-]+ray\b": "xray",
    r"\bnine\s+er\b": "niner",
    r"\balfa\b": "alpha",
}


def normalize_transcript(text: str) -> str:
    normalized = text.lower()
    for pattern, replacement in REPLACEMENTS.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Whisper sometimes hears the N-number prefix as "number".
    normalized = re.sub(
        (
            r"\bnumber (?=(?:zero|oh|one|wun|two|too|three|tree|four|fower|five|fife|six|"
            r"seven|eight|nine|niner)\b)"
        ),
        "november ",
        normalized,
    )
    return normalized
