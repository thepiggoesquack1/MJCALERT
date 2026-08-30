from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz.fuzz import ratio

from mry_alert.config import (
    DestinationConfig,
    IntentDetectionConfig,
    KnownDestinationConfig,
)
from mry_alert.models import DestinationIntentCategory


@dataclass(frozen=True)
class ParsedDestination:
    canonical_name: str
    alias: str
    confidence: float
    start: int
    end: int


@dataclass(frozen=True)
class DestinationEvidence:
    detected: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)
    exact_destination: bool = False
    has_taxi_context: bool = False
    intent_category: DestinationIntentCategory = DestinationIntentCategory.NONE
    route_cues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CorrectionEvidence:
    detected: bool
    confidence: float
    corrected_destination: str | None = None
    reasons: list[str] = field(default_factory=list)


def parse_destination(
    text: str, destinations: list[KnownDestinationConfig]
) -> ParsedDestination | None:
    """Return the most recently stated exact known destination."""
    matches: list[ParsedDestination] = []
    for destination in destinations:
        for alias in destination.aliases:
            pattern = rf"\b{re.escape(alias.lower())}\b"
            for match in re.finditer(pattern, text.lower()):
                confidence = 0.99 if alias.lower() == destination.canonical_name.lower() else 0.96
                matches.append(
                    ParsedDestination(
                        canonical_name=destination.canonical_name,
                        alias=alias,
                        confidence=confidence,
                        start=match.start(),
                        end=match.end(),
                    )
                )
    if not matches:
        return None
    return max(matches, key=lambda item: (item.end, len(item.alias)))


def detect_correction(
    text: str,
    destination: ParsedDestination | None,
    previous_destination: str,
) -> CorrectionEvidence:
    normalized = text.lower()
    different_destination = bool(
        destination and destination.canonical_name.lower() != previous_destination.lower()
    )
    strong_patterns = {
        "correction": r"\bcorrection\b",
        "negative": r"\bnegative\b",
        "scratch that": r"\bscratch(?: that| [a-z ]{1,40})?\b",
        "disregard": r"\bdisregard\b",
        "actually": r"\bactually\b",
        "wait": r"\bwait\b(?!\s+no)",
        "wait no": r"\bwait\s+no\b",
        "make that": r"\bmake that\b",
        "instead": r"\binstead\b",
        "change that to": r"\bchange that to\b",
    }
    matched = [
        label for label, pattern in strong_patterns.items() if re.search(pattern, normalized)
    ]
    # A bare "no" is only corrective when it introduces a different explicit destination.
    bare_no = bool(re.search(r"(?:^|\s)no(?:\s|$)", normalized)) and different_destination
    redirect = bool(re.search(r"\b(?:going to|parking at)\b", normalized)) and different_destination
    if destination and different_destination and (matched or bare_no or redirect):
        reasons = [f'correction phrase "{label}"' for label in matched]
        if bare_no and "wait no" not in matched:
            reasons.append("contextual no before a different explicit destination")
        if redirect:
            reasons.append("different destination stated with going/parking context")
        reasons.append(f'new explicit destination "{destination.canonical_name}"')
        return CorrectionEvidence(
            True, 0.98 if matched else 0.9, destination.canonical_name, reasons
        )
    cancellation_markers = {"correction", "negative", "scratch that", "disregard", "wait no"}
    matched_cancellation = cancellation_markers.intersection(matched)
    same_destination = bool(
        destination and destination.canonical_name.lower() == previous_destination.lower()
    )
    standalone_cancellation = bool({"scratch that", "disregard"}.intersection(matched))
    if matched_cancellation and (
        same_destination or previous_destination.lower() in normalized or standalone_cancellation
    ):
        return CorrectionEvidence(
            True,
            0.9,
            None,
            [f'correction phrase "{matched[0]}"', "previous destination explicitly negated"],
        )
    return CorrectionEvidence(False, 0.0)


def detect_destination(
    text: str,
    config: DestinationConfig,
    responding_to_parking: bool = False,
    authorized_ground_source: bool = False,
    route_intent_threshold: float = 0.80,
    fuzzy_intent_matching: bool = True,
    fuzzy_intent_minimum_score: int = 85,
    intent_config: IntentDetectionConfig | None = None,
) -> DestinationEvidence:
    intent = intent_config or IntentDetectionConfig(route_context_threshold=route_intent_threshold)
    exact = next((phrase for phrase in config.phrases if phrase in text), None)
    full_exact = exact == "monterey jet center"
    if exact and not full_exact and not intent.allow_partial_jet_center_match:
        exact = None
    taxi = next((phrase for phrase in config.taxi_context_phrases if phrase in text), None)
    reasons: list[str] = []
    if exact:
        reasons.append(f'exact destination phrase "{exact}"')
    fuzzy = False
    if not exact:
        windows = [" ".join(text.split()[i : i + 3]) for i in range(max(1, len(text.split()) - 2))]
        fuzzy = fuzzy_intent_matching and any(
            ratio(window, "monterey jet center") >= fuzzy_intent_minimum_score for window in windows
        )
        if fuzzy:
            reasons.append("strong fuzzy destination phrase")
    if taxi:
        reasons.append(f'taxi/parking context "{taxi}"')
    if responding_to_parking and (exact or fuzzy):
        reasons.append("response to recent parking question")
    explicit_taxi = bool(
        re.search(
            r"\b(?:request\s+)?taxi(?:\s+(?:to|for)|\b.*\b(?:jet|destination))",
            text,
        )
    )
    parking_statement = bool(
        re.search(
            r"\b(?:parking(?:\s+at)?|going\s+to|(?:we\s+)?would\s+like\s+to\s+go\s+to|"
            r"headed\s+(?:to|for)|"
            r"destination\s+is|request(?:ing)?(?:\s+to)?)\b.*"
            r"\b(?:monterey\s+jet(?:\s+center)?|(?:the\s+)?jet\s+center)\b",
            text,
        )
    )
    route_patterns = {
        "phonetic taxiway": (
            r"\b(?:alpha|bravo|charlie|delta|echo|foxtrot|golf|hotel|juliet|kilo|"
            r"lima|mike)\b"
        ),
        "runway": r"\brunway\b",
        "turn": r"\bturn\s+(?:left|right)\b|\bleft\s+at\b|\bright\s+at\b",
        "continue": r"\bcontinue\b",
        "proceed": r"\bproceed\b",
        "via": r"\bvia\b",
        "then": r"\bthen\b",
        "to": r"\bto\b",
        "cross": r"\bcross\b",
        "hold short": r"\bhold\s+short\b",
        "monitor ground": r"\bmonitor\s+ground\b",
        "ramp": r"\bramp\b",
    }
    route_cues = [label for label, pattern in route_patterns.items() if re.search(pattern, text)]
    route_score = 0.0
    if taxi:
        route_score += 0.45
    if "via" in route_cues:
        route_score += 0.4
    if "phonetic taxiway" in route_cues:
        route_score += 0.25
        if "to" in route_cues:
            route_score += 0.4
    if any(
        cue in route_cues
        for cue in (
            "turn",
            "continue",
            "proceed",
            "cross",
            "hold short",
            "monitor ground",
            "runway",
        )
    ):
        route_score += 0.4
    if authorized_ground_source and route_cues:
        route_score += 0.1
    route_connected = bool(
        exact
        and authorized_ground_source
        and route_cues
        and route_score >= intent.route_context_threshold
    )
    if exact and explicit_taxi:
        return DestinationEvidence(
            True,
            0.98 if full_exact else 0.91,
            reasons,
            True,
            True,
            DestinationIntentCategory.EXPLICIT_TAXI_REQUEST,
            route_cues,
        )
    if exact and parking_statement:
        return DestinationEvidence(
            True,
            0.97 if full_exact else 0.9,
            reasons,
            True,
            bool(taxi),
            DestinationIntentCategory.EXPLICIT_PARKING_STATEMENT,
            route_cues,
        )
    route_confidence = (
        0.96
        if full_exact
        else max(
            intent.destination_phrase_threshold,
            intent.weak_phrase_with_strong_route_threshold + 0.16,
        )
    )
    if exact and route_connected and route_confidence >= intent.destination_phrase_threshold:
        return DestinationEvidence(
            True,
            route_confidence,
            [*reasons, "ground routing explicitly leads to destination"],
            True,
            True,
            DestinationIntentCategory.GROUND_ROUTE_TO_DESTINATION,
            route_cues,
        )
    if exact and responding_to_parking:
        return DestinationEvidence(
            True,
            0.95,
            reasons,
            True,
            False,
            DestinationIntentCategory.PARKING_PROMPT_RESPONSE,
            route_cues,
        )
    if fuzzy and (
        taxi
        or responding_to_parking
        or (
            intent.allow_contextual_destination_inference
            and route_score >= intent.route_context_threshold
        )
    ):
        category = (
            DestinationIntentCategory.PARKING_PROMPT_RESPONSE
            if responding_to_parking
            else DestinationIntentCategory.EXPLICIT_TAXI_REQUEST
        )
        return DestinationEvidence(
            True,
            max(0.82, intent.weak_phrase_with_strong_route_threshold),
            reasons,
            False,
            bool(taxi),
            category,
            route_cues,
        )
    return DestinationEvidence(
        False,
        0.45 if exact else 0.0,
        reasons,
        bool(exact),
        bool(taxi),
        DestinationIntentCategory.WEAK_DESTINATION_MENTION
        if exact
        else DestinationIntentCategory.NONE,
        route_cues,
    )
