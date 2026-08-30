from __future__ import annotations

import re

from mry_alert.models import SpokenCallsign

PHONETIC = {
    "alpha": "A",
    "alfa": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliet": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "xray": "X",
    "yankee": "Y",
    "zulu": "Z",
}
DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "wun": "1",
    "two": "2",
    "too": "2",
    "three": "3",
    "tree": "3",
    "four": "4",
    "fower": "4",
    "five": "5",
    "fife": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "niner": "9",
}
AIRCRAFT_TYPES = {
    "citation",
    "cessna",
    "cherokee",
    "bonanza",
    "gulfstream",
    "falcon",
    "pilatus",
    "cirrus",
    "kingair",
    "king air",
}
BOUNDARY_WORDS = {
    "request",
    "taxi",
    "say",
    "parking",
    "clear",
    "cleared",
    "ready",
    "with",
    "at",
    "to",
    "for",
    "where",
}


def _valid_n_number(value: str) -> bool:
    # FAA-like shape: N + 1-5 characters; first is digit, at most two trailing letters.
    return bool(re.fullmatch(r"N[1-9][0-9]{0,3}[A-Z]{0,2}", value)) and len(value) <= 6


def parse_callsign(text: str) -> SpokenCallsign | None:
    words = text.lower().replace("x-ray", "xray").split()
    best: SpokenCallsign | None = None
    for start, word in enumerate(words):
        prefix: str | None = None
        explicit_n = word in {"november", "n"}
        if word in AIRCRAFT_TYPES or (
            start + 1 < len(words) and f"{word} {words[start + 1]}" in AIRCRAFT_TYPES
        ):
            prefix = word.title()
        if not explicit_n and prefix is None and word not in DIGITS:
            continue
        chars: list[str] = []
        index = start + (1 if explicit_n or prefix else 0)
        while index < len(words) and len(chars) < 6:
            token = words[index]
            if token in DIGITS:
                chars.append(DIGITS[token])
            elif token in PHONETIC and token != "november":
                chars.append(PHONETIC[token])
            elif len(token) == 1 and token.isdigit():
                chars.append(token)
            elif token in BOUNDARY_WORDS:
                break
            else:
                break
            index += 1
        joined = "".join(chars)
        if len(joined) < 2 or not joined[0].isdigit() or not any(c.isalpha() for c in joined):
            continue
        full = f"N{joined}" if explicit_n and _valid_n_number(f"N{joined}") else None
        confidence = 0.98 if full else (0.85 if prefix else 0.75)
        reasons = (
            ["explicit November registration prefix"]
            if full
            else ["abbreviated digit-and-letter aviation callsign"]
        )
        if prefix:
            reasons.append(f"aircraft type prefix {prefix}")
        candidate = SpokenCallsign(
            original_text=" ".join(words[start:index]),
            normalized_form=full or joined,
            full_registration=full,
            suffix=None if full else joined,
            aircraft_type_prefix=prefix,
            parse_confidence=confidence,
            parse_reasons=reasons,
        )
        if best is None or candidate.parse_confidence > best.parse_confidence:
            best = candidate
    return best
