from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.destination import detect_correction, parse_destination
from mry_alert.detection.normalizer import normalize_transcript


def _distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = normalize_transcript(reference).split()
    actual = normalize_transcript(hypothesis).split()
    return _distance(expected, actual) / max(1, len(expected))


def character_error_rate(reference: str, hypothesis: str) -> float:
    expected = list(normalize_transcript(reference).replace(" ", ""))
    actual = list(normalize_transcript(hypothesis).replace(" ", ""))
    return _distance(expected, actual) / max(1, len(expected))


@dataclass(frozen=True)
class AviationEntities:
    registration: str | None = None
    operator_callsigns: list[str] = field(default_factory=list)
    runways: list[str] = field(default_factory=list)
    taxiways: list[str] = field(default_factory=list)
    destination: str | None = None
    intent: str | None = None
    correction: bool = False


def extract_aviation_entities(text: str, destinations: list[Any]) -> AviationEntities:
    normalized = normalize_transcript(text)
    callsign = parse_callsign(normalized)
    destination = parse_destination(normalized, destinations)
    runway_matches = re.findall(
        r"\brunway\s+((?:[a-z0-9]+\s*){1,4}?)(?=\s+(?:left|right|center|taxi|via|hold|$)|$)",
        normalized,
    )
    taxiway_names = re.findall(
        r"\b(?:taxiway|via|at)\s+(alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|"
        r"juliet|kilo|lima|mike)\b",
        normalized,
    )
    operators = re.findall(r"\b(?:skywest|united|american|southwest|alaska)\s+\d+\b", normalized)
    intent = None
    if re.search(r"\b(?:request\s+taxi|taxi\s+to|going\s+to|parking\s+at)\b", normalized):
        intent = "taxi_or_parking"
    elif re.search(r"\b(?:hold short|cross|taxi via|turn (?:left|right))\b", normalized):
        intent = "ground_route"
    correction = detect_correction(normalized, destination, "Monterey Jet Center").detected
    return AviationEntities(
        registration=callsign.full_registration if callsign else None,
        operator_callsigns=operators,
        runways=[item.strip() for item in runway_matches],
        taxiways=taxiway_names,
        destination=destination.canonical_name if destination else None,
        intent=intent,
        correction=correction,
    )


def aviation_entity_scores(
    expected: AviationEntities, actual: AviationEntities
) -> dict[str, float]:
    def exact(left: object, right: object) -> float:
        return float(left == right)

    return {
        "callsign_accuracy": exact(expected.registration, actual.registration),
        "operator_callsign_accuracy": exact(expected.operator_callsigns, actual.operator_callsigns),
        "destination_accuracy": exact(expected.destination, actual.destination),
        "runway_accuracy": exact(expected.runways, actual.runways),
        "taxiway_accuracy": exact(expected.taxiways, actual.taxiways),
        "intent_accuracy": exact(expected.intent, actual.intent),
        "correction_accuracy": exact(expected.correction, actual.correction),
    }


def route_term_accuracy(reference: str, hypothesis: str) -> float:
    vocabulary = {
        "runway",
        "taxiway",
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "hold",
        "short",
        "cross",
        "left",
        "right",
        "via",
    }
    expected = {word for word in normalize_transcript(reference).split() if word in vocabulary}
    actual = {word for word in normalize_transcript(hypothesis).split() if word in vocabulary}
    return len(expected & actual) / max(1, len(expected))


def classify_rtf(
    rtf: float,
    excellent: float = 0.5,
    acceptable: float = 0.8,
    risky: float = 1.0,
) -> str:
    if rtf <= excellent:
        return "excellent"
    if rtf <= acceptable:
        return "acceptable"
    if rtf <= risky:
        return "risky"
    return "unsuitable for live use"


def load_expected_transcript(path: Path | None) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path else None


def load_expected_json(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def evaluation_metrics(
    reference: str, hypothesis: str, destinations: list[Any]
) -> dict[str, object]:
    expected_entities = extract_aviation_entities(reference, destinations)
    actual_entities = extract_aviation_entities(hypothesis, destinations)
    return {
        "word_error_rate": word_error_rate(reference, hypothesis),
        "character_error_rate": character_error_rate(reference, hypothesis),
        "route_taxiway_term_accuracy": route_term_accuracy(reference, hypothesis),
        "expected_entities": asdict(expected_entities),
        "actual_entities": asdict(actual_entities),
        **aviation_entity_scores(expected_entities, actual_entities),
    }
