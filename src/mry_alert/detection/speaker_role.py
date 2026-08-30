from __future__ import annotations

import re
from dataclasses import dataclass

from mry_alert.detection.callsign import AIRCRAFT_TYPES, DIGITS, PHONETIC
from mry_alert.models import SpeakerRole, SpokenCallsign


@dataclass(frozen=True)
class SpeakerRoleEvidence:
    role: SpeakerRole
    confidence: float
    reasons: list[str]


def _callsign_count(text: str) -> int:
    prefixes = sorted({"november", "n", *AIRCRAFT_TYPES}, key=len, reverse=True)
    symbols = sorted({*DIGITS, *PHONETIC}, key=len, reverse=True)
    pattern = (
        rf"\b(?:{'|'.join(re.escape(item) for item in prefixes)})\s+"
        rf"(?:{'|'.join(re.escape(item) for item in symbols)})(?:\s+(?:"
        rf"{'|'.join(re.escape(item) for item in symbols)})){{1,5}}\b"
    )
    return len(re.findall(pattern, text))


def infer_speaker_role(
    text: str,
    callsign: SpokenCallsign | None,
    *,
    responding_to_destination_prompt: bool = False,
) -> SpeakerRoleEvidence:
    """Infer a likely radio role from words and turn-taking, never voice identity."""
    controller_score = 0.0
    pilot_score = 0.0
    controller_reasons: list[str] = []
    pilot_reasons: list[str] = []

    pilot_patterns = {
        "first-person taxi request": r"\brequest(?:ing)?\s+taxi\b",
        "first-person clearance request": r"\brequest(?:ing)?\s+(?:ifr\s+)?clearance\b",
        "first-person parking statement": r"\bwe(?:\s+(?:are|re))?\s+parking\s+at\b",
        "first-person destination statement": (
            r"\b(?:we(?:\s+(?:are|re))?\s+going\s+to|"
            r"we\s+(?:would|d)\s+like\s+to\s+(?:go|taxi|head|proceed)(?:\s+over)?\s+to|"
            r"we\s+only\s+go\s+to)\b"
        ),
        "ready-to-taxi report": r"\bready\s+to\s+taxi\b",
        "holding-short report": r"\bholding\s+short\b",
    }
    matched_pilot = [
        reason for reason, pattern in pilot_patterns.items() if re.search(pattern, text)
    ]
    if matched_pilot:
        pilot_score += 0.6 + min(0.15, 0.05 * (len(matched_pilot) - 1))
        pilot_reasons.extend(matched_pilot)

    controller_patterns = {
        "hold-short command": r"\bhold\s+short\b",
        "frequency/contact command": r"\bcontact\b|\bremain\s+this\s+frequency\b",
        "runway-crossing command": r"\bcross\b",
        "line-up command": r"\bline\s+up\b",
        "clearance command": r"\bcleared\b",
        "parking query": r"\bsay\s+parking\b|\bwhere\s+are\s+you\s+parking\b",
        "advise command": r"\badvise\b",
        "continue command": r"\bcontinue(?:\s+down)?\b",
        "turn command": r"\bturn\s+(?:left|right)\b|\bleft\s+turn\b|\bright\s+turn\b",
        "exit instruction": r"\b(?:call|report)\s+(?:your\s+)?exit\b|\bi(?:\s+will|\s+ll)\s+call\b",
    }
    matched_controller = [
        reason for reason, pattern in controller_patterns.items() if re.search(pattern, text)
    ]
    explicit_pilot_language = bool(matched_pilot)
    taxi_command = bool(re.search(r"\btaxi\b", text)) and not explicit_pilot_language
    if taxi_command:
        matched_controller.append("taxi command without request language")
    if matched_controller:
        controller_score += 0.45 + min(0.15, 0.05 * (len(matched_controller) - 1))
        controller_reasons.extend(matched_controller)

    callsign_text = callsign.original_text if callsign else ""
    callsign_at_start = bool(callsign_text and text.startswith(callsign_text))
    callsign_at_end = bool(callsign_text and text.endswith(callsign_text))
    callsign_position = text.find(callsign_text) if callsign_text else -1
    callsign_near_start = bool(callsign_position > 0 and len(text[:callsign_position].split()) <= 2)
    callsign_only = bool(callsign_text and text.strip() == callsign_text.strip())
    if callsign_at_start and not callsign_only and not explicit_pilot_language:
        controller_score += 0.3
        controller_reasons.append("transmission begins by addressing a callsign")
    elif callsign_near_start and not explicit_pilot_language:
        controller_score += 0.25
        controller_reasons.append("transmission clearly addresses a callsign near the beginning")
    if callsign_at_end and not callsign_at_start:
        pilot_score += 0.3
        pilot_reasons.append("callsign readback appears at the end")
    elif callsign_only:
        pilot_score += 0.55
        pilot_reasons.append("short callsign-only readback")

    callsign_count = _callsign_count(text)
    if callsign_count >= 2:
        controller_score += 0.35
        controller_reasons.append("multiple aircraft callsigns addressed in sequence")

    if responding_to_destination_prompt:
        pilot_score += 0.7
        pilot_reasons.append("short destination reply follows an unambiguous controller prompt")

    first_person = bool(re.search(r"\b(?:we|we're|we are|request|requesting)\b", text))
    if controller_score and not first_person:
        controller_score += 0.1
        controller_reasons.append("no first-person request language")

    difference = abs(controller_score - pilot_score)
    if controller_score < 0.45 and pilot_score < 0.45:
        return SpeakerRoleEvidence(SpeakerRole.UNKNOWN, 0.0, [])
    if controller_score and pilot_score and difference < 0.2:
        return SpeakerRoleEvidence(
            SpeakerRole.UNKNOWN,
            min(0.65, max(controller_score, pilot_score)),
            ["controller and pilot phraseology conflict"],
        )
    if controller_score > pilot_score:
        confidence = min(0.98, 0.55 + controller_score * 0.45)
        return SpeakerRoleEvidence(SpeakerRole.CONTROLLER, confidence, controller_reasons)
    confidence = min(0.98, 0.55 + pilot_score * 0.45)
    return SpeakerRoleEvidence(SpeakerRole.PILOT, confidence, pilot_reasons)
