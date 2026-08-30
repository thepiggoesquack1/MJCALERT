from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from mry_alert.models import AircraftMatch


class OutboundDisposition(StrEnum):
    ARRIVAL_ELIGIBLE = "arrival_eligible"
    OUTBOUND_DEPARTURE_FILTERED = "outbound_departure_filtered"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class OutboundInference:
    disposition: OutboundDisposition
    reasons: list[str]


_JET_CENTER = (
    r"(?:monterey\s+jet(?:\s+cent(?:er|re))?|jet\s+cent(?:er|re)|"
    r"monterey\s+cent(?:er|re)|(?:monterey\s+)?jet)"
)
_CLEAR_INBOUND = re.compile(
    r"\b(?:taxi(?:ing)?\s+(?:via\s+[\w\s]+\s+)?to|going\s+to|"
    r"(?:we\s+)?would\s+like\s+to\s+go\s+to|headed\s+(?:to|for)|"
    rf"route\s+to|proceed(?:ing)?\s+to)\s+(?:the\s+)?{_JET_CENTER}\b"
)
_OUTBOUND_REQUEST = re.compile(
    r"\b(?:request(?:ing)?\s+taxi|ready\s+to\s+taxi|taxi\s+for\s+departure|"
    r"taxi\s+to\s+(?:runway|departure)|taxi\s+via\s+[a-z0-9\s]+|departing|"
    r"departure\s+request|from\s+(?:the\s+)?(?:monterey\s+)?jet)\b"
)
_CURRENT_LOCATION = re.compile(
    rf"\b(?:we(?:\s+(?:are|re))?\s+)?(?:currently\s+)?(?:at|parking\s+at|parked\s+at|"
    rf"out\s+of|from|departing)\s+(?:the\s+)?{_JET_CENTER}\b"
)
_PARKING_INFORMATION = re.compile(
    rf"\b(?:parking(?:\s+at)?|parked(?:\s+at)?|at|out\s+of|from|departing)\s+"
    rf"(?:the\s+)?{_JET_CENTER}\b"
)


def infer_outbound_departure(text: str, match: AircraftMatch) -> OutboundInference:
    """Infer direction from phraseology and ADS-B history, never from voice identity."""
    normalized = " ".join(text.lower().split())
    if _CLEAR_INBOUND.search(normalized):
        return OutboundInference(
            OutboundDisposition.ARRIVAL_ELIGIBLE,
            ["transmission explicitly describes travel to Monterey Jet Center"],
        )
    if match.recently_landed:
        return OutboundInference(
            OutboundDisposition.ARRIVAL_ELIGIBLE,
            ["ADS-B history indicates the aircraft recently landed"],
        )
    if match.movement_state == "moving_toward_jet_center":
        return OutboundInference(
            OutboundDisposition.ARRIVAL_ELIGIBLE,
            ["ADS-B history shows movement toward Monterey Jet Center"],
        )

    parking_information = bool(_PARKING_INFORMATION.search(normalized))
    outbound_request = bool(_OUTBOUND_REQUEST.search(normalized))
    current_location = bool(_CURRENT_LOCATION.search(normalized))
    inside = match.inside_fbo_geofence
    history_supports_departure = (
        match.was_stationary_at_fbo
        or match.moving_away_from_destination
        or match.movement_state == "parked_at_jet_center"
    )
    if (
        (inside or history_supports_departure)
        and parking_information
        and outbound_request
        and current_location
    ):
        reasons = [
            "parking statement describes the aircraft's current location",
            "pilot phraseology requests taxi for departure",
        ]
        if inside:
            reasons.insert(
                0,
                "already inside "
                + (", ".join(match.fbo_geofence_names) or "a configured FBO")
                + " geofence before transmission",
            )
        if match.was_stationary_at_fbo:
            reasons.append("ADS-B history shows the aircraft was stationary or parked there")
        elif match.movement_state == "parked_at_jet_center":
            reasons.append("current ADS-B state shows the aircraft parked at the ramp")
        if match.moving_away_from_destination:
            reasons.append("ADS-B history shows taxiing away from the ramp")
        return OutboundInference(
            OutboundDisposition.OUTBOUND_DEPARTURE_FILTERED,
            reasons,
        )

    if (inside and parking_information) or (current_location and parking_information):
        reasons = [
            "parking wording describes or may describe the aircraft's current location",
            "movement and departure evidence was not strong enough to suppress",
        ]
        if inside:
            reasons.insert(0, "aircraft is already inside a configured FBO geofence")
        if outbound_request:
            reasons.append("departure phraseology was present but ADS-B support was incomplete")
        return OutboundInference(
            OutboundDisposition.UNRESOLVED,
            reasons,
        )
    return OutboundInference(
        OutboundDisposition.ARRIVAL_ELIGIBLE,
        ["no strong evidence that the parking statement describes an outbound aircraft"],
    )
