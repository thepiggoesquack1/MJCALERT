from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from mry_alert.models import DestinationState, SpeakerRole, SpokenCallsign


@dataclass
class RadioContact:
    callsign: SpokenCallsign
    timestamp: datetime
    awaiting_parking: bool = False
    awaiting_parking_since: datetime | None = None
    destination_state: DestinationState = DestinationState.UNKNOWN
    destination: str | None = None
    route_cues: list[str] = field(default_factory=list)
    pilot_arrival_evidence_at: datetime | None = None


class ConversationContext:
    def __init__(self, window_seconds: float = 120) -> None:
        self.window = timedelta(seconds=window_seconds)
        self._contacts: dict[str, RadioContact] = {}

    def _expire(self, now: datetime) -> None:
        self._contacts = {
            key: value
            for key, value in self._contacts.items()
            if now - value.timestamp <= self.window
        }

    def contact_count(self, timestamp: datetime) -> int:
        self._expire(timestamp)
        return len(self._contacts)

    def observe(
        self,
        callsign: SpokenCallsign | None,
        text: str,
        timestamp: datetime,
        speaker_role: SpeakerRole = SpeakerRole.UNKNOWN,
        speaker_role_confidence: float = 0.0,
    ) -> None:
        self._expire(timestamp)
        if callsign:
            key = callsign.normalized_form
            existing = self._contacts.get(key)
            route_cues: list[str] = []
            route_phrases = {
                "taxi": ("taxi",),
                "parking": ("parking", "park"),
                "via": ("via",),
                "turn": ("turn left", "turn right", "left at", "right at"),
                "monitor ground": ("monitor ground",),
            }
            for cue, phrases in route_phrases.items():
                if any(phrase in text for phrase in phrases):
                    route_cues.append(cue)
            if route_cues and any(
                taxiway in text
                for taxiway in (
                    "alpha",
                    "bravo",
                    "charlie",
                    "delta",
                    "echo",
                    "foxtrot",
                    "golf",
                    "hotel",
                )
            ):
                route_cues.append("taxiway")
            destination_prompt = any(
                phrase in text
                for phrase in (
                    "say parking",
                    "where are you parking",
                    "advise parking",
                    "say destination",
                    "state destination",
                    "where are you going",
                )
            )
            awaiting = (
                speaker_role == SpeakerRole.CONTROLLER
                and speaker_role_confidence >= 0.7
                and destination_prompt
            )
            pilot_arrival_evidence = bool(
                speaker_role == SpeakerRole.PILOT
                and speaker_role_confidence >= 0.55
                and re.search(
                    r"\b(?:request|would like|we d like|like to|we (?:are|re) going to|"
                    r"headed to|taxi(?:ing)? to|"
                    r"parking at|for parking)\b",
                    text,
                )
                and re.search(r"\b(?:monterey\s+)?jet(?:\s+center)?\b", text)
            )
            if existing:
                existing.callsign = callsign
                existing.timestamp = timestamp
                existing.awaiting_parking = existing.awaiting_parking or awaiting
                if awaiting:
                    existing.awaiting_parking_since = timestamp
                existing.route_cues = list(dict.fromkeys([*existing.route_cues, *route_cues]))
                if pilot_arrival_evidence:
                    existing.pilot_arrival_evidence_at = timestamp
            else:
                self._contacts[key] = RadioContact(
                    callsign,
                    timestamp,
                    awaiting,
                    timestamp if awaiting else None,
                    route_cues=route_cues,
                    pilot_arrival_evidence_at=(timestamp if pilot_arrival_evidence else None),
                )

    def parking_response_contact(self, timestamp: datetime) -> tuple[SpokenCallsign | None, bool]:
        self._expire(timestamp)
        waiting = [contact for contact in self._contacts.values() if contact.awaiting_parking]
        if len(waiting) == 1:
            selected = waiting[0]
            competing_newer = any(
                contact.callsign.normalized_form != selected.callsign.normalized_form
                and selected.awaiting_parking_since is not None
                and contact.timestamp > selected.awaiting_parking_since
                for contact in self._contacts.values()
            )
            if not competing_newer:
                selected.awaiting_parking = False
                selected.awaiting_parking_since = None
                return selected.callsign, False
            return None, True
        return None, len(waiting) > 1

    def contact(self, callsign: SpokenCallsign, timestamp: datetime) -> RadioContact | None:
        self._expire(timestamp)
        return self._contacts.get(callsign.normalized_form)

    def unique_recent_contact(self, timestamp: datetime) -> RadioContact | None:
        self._expire(timestamp)
        if len(self._contacts) != 1:
            return None
        return next(iter(self._contacts.values()))

    def route_cues(self, callsign: SpokenCallsign | None, timestamp: datetime) -> list[str]:
        self._expire(timestamp)
        if callsign:
            contact = self._contacts.get(callsign.normalized_form)
        else:
            contact = self.unique_recent_contact(timestamp)
        return list(contact.route_cues) if contact else []

    def has_pilot_arrival_evidence(
        self, callsign: SpokenCallsign | None, timestamp: datetime
    ) -> bool:
        self._expire(timestamp)
        contact = self.contact(callsign, timestamp) if callsign else None
        return bool(
            contact
            and contact.pilot_arrival_evidence_at
            and timestamp - contact.pilot_arrival_evidence_at <= self.window
        )

    def set_destination_state(
        self,
        contact_key: str,
        state: DestinationState,
        destination: str | None,
        timestamp: datetime,
    ) -> None:
        self._expire(timestamp)
        contact = self._contacts.get(contact_key)
        if contact:
            contact.destination_state = state
            contact.destination = destination
            contact.timestamp = timestamp
