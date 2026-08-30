from __future__ import annotations

from collections import OrderedDict
from datetime import datetime

from mry_alert.models import (
    AlertEvent,
    AlertEventType,
    ConfirmationStatus,
    DetectionDecision,
    MatchStatus,
    OperatorAcknowledgement,
    PendingDestinationEvent,
    SessionEventRecord,
    TranscriptEvent,
)


def _classifier_confidence(event: TranscriptEvent) -> float | None:
    result = event.audio_intent
    if result is None:
        return None
    return max(
        result.destination_confidence,
        result.intent_confidence,
        result.correction_confidence,
    )


def _transcript_transition(event: TranscriptEvent) -> str:
    reasons = " ".join(event.detection_reasons).lower()
    if event.traffic_filter_decision == "ignored_scheduled_airline":
        return "airline_filtered"
    if "outbound_departure_filtered" in reasons or (
        "current location" in reasons and event.detection_decision == DetectionDecision.IGNORED
    ):
        return "outbound_filtered"
    if "duplicate_suppressed" in reasons or "duplicate" in reasons:
        return "duplicate"
    if "suppressed" in reasons or "active destination event already exists" in reasons:
        return "suppressed"
    return {
        DetectionDecision.CONFIRMED: "confirmed",
        DetectionDecision.PENDING: "pending",
        DetectionDecision.CORRECTED: "corrected",
        DetectionDecision.CANCELLED: "cancelled",
        DetectionDecision.UNRESOLVED: "unresolved",
        DetectionDecision.AMBIGUOUS: "ambiguous",
        DetectionDecision.IGNORED: "denied",
    }.get(event.detection_decision, "denied")


def _direction(transition: str) -> str:
    if transition == "outbound_filtered":
        return "outbound"
    if transition in {"ambiguous", "unresolved", "possible"}:
        return "ambiguous"
    if transition in {"confirmed", "pending", "corrected", "cancelled"}:
        return "inbound"
    return "denied"


def record_transcript(event: TranscriptEvent) -> SessionEventRecord:
    transition = _transcript_transition(event)
    return SessionEventRecord(
        event_id=event.event_id,
        transition_type=transition,
        timestamp=event.timestamp,
        registration=event.identified_registration,
        spoken_callsign=event.detected_callsign,
        operator_name=event.identified_operator,
        aircraft_type=event.aircraft_type_name or event.aircraft_type_code,
        aircraft_type_code=event.aircraft_type_code,
        aircraft_type_name=event.aircraft_type_name,
        manufacturer=event.manufacturer,
        model=event.model,
        aircraft_category=event.aircraft_category,
        aircraft_type_source=event.aircraft_type_source,
        aircraft_type_confidence=event.aircraft_type_confidence,
        destination=event.destination_candidate,
        intent=event.intent_category.value,
        direction_state=_direction(transition),
        adsb_movement_state=event.adsb_movement_state,
        adsb_score=event.adsb_winning_score,
        winning_margin=event.adsb_winning_margin,
        final_decision=event.detection_decision.value,
        notification_status="suppressed"
        if transition in {"suppressed", "duplicate", "outbound_filtered", "airline_filtered"}
        else "not_sent",
        decision_reasons=[*event.detection_reasons, *event.traffic_filter_reasons],
        transcript_excerpt=event.text,
        classifier_confidence=_classifier_confidence(event),
        whisper_confidence=event.transcription_confidence,
        decoder_confidence=event.whisper_quality,
    )


def record_detection_event(
    event: AlertEvent | PendingDestinationEvent,
) -> SessionEventRecord:
    if event.confirmation_status == ConfirmationStatus.EXPIRED:
        transition = "expired"
    elif event.confirmation_status == ConfirmationStatus.UNRESOLVED:
        transition = "unresolved"
    elif event.event_type == AlertEventType.DESTINATION_CORRECTION:
        transition = "corrected"
    elif event.event_type == AlertEventType.DESTINATION_CANCELLED:
        transition = "cancelled"
    elif isinstance(event, PendingDestinationEvent):
        transition = "pending"
    elif event.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED, MatchStatus.LIKELY}:
        transition = "possible"
    else:
        transition = "confirmed"
    notification_status = {
        "pending": "pending",
        "possible": "possible",
        "confirmed": "confirmed",
        "corrected": "corrected",
        "cancelled": "cancelled",
        "expired": "expired",
        "unresolved": "unresolved",
    }[transition]
    return SessionEventRecord(
        event_id=event.event_id,
        transition_type=transition,
        timestamp=event.timestamp,
        registration=event.registration,
        spoken_callsign=event.spoken_callsign,
        operator_name=event.operator_name,
        aircraft_type=event.aircraft_type,
        aircraft_type_code=event.aircraft_type_code,
        aircraft_type_name=event.aircraft_type_name,
        manufacturer=event.manufacturer,
        model=event.model,
        aircraft_category=event.aircraft_category,
        aircraft_type_source=event.aircraft_type_source,
        aircraft_type_confidence=event.aircraft_type_confidence,
        destination=event.corrected_destination or event.destination,
        intent=event.intent.value,
        direction_state=_direction(transition),
        adsb_movement_state=event.adsb_movement_state,
        adsb_score=event.adsb_winning_score,
        winning_margin=event.adsb_winning_margin,
        final_decision=transition,
        notification_status=notification_status,
        chrome_delivery_result="pending_not_notified"
        if transition == "pending"
        else "not_attempted",
        decision_reasons=event.match_reasons,
        transcript_excerpt=event.transcript_excerpt,
        classifier_confidence=event.classifier_confidence,
        whisper_confidence=event.whisper_confidence,
        decoder_confidence=event.decoder_confidence,
        original_event_id=event.original_event_id,
        test=event.test if isinstance(event, AlertEvent) else False,
    )


class SessionEventHistory:
    """Bounded process-memory-only event history with no filesystem dependency."""

    def __init__(self, maximum: int = 1000) -> None:
        self.maximum = maximum
        self._records: OrderedDict[tuple[str, str], SessionEventRecord] = OrderedDict()

    def append(self, record: SessionEventRecord) -> None:
        # A terminal transition replaces the provisional row instead of leaving
        # an event looking pending forever.
        if record.transition_type != "pending":
            self._records.pop((record.event_id, "pending"), None)
        key = (record.event_id, record.transition_type)
        if key in self._records:
            self._records.pop(key)
        self._records[key] = record
        while len(self._records) > self.maximum:
            self._records.popitem(last=False)

    def recent(self) -> list[SessionEventRecord]:
        return list(reversed(self._records.values()))

    def update_delivery(
        self,
        event_id: str,
        *,
        chrome: str | None = None,
        ntfy: str | None = None,
    ) -> None:
        matching = [key for key in self._records if key[0] == event_id]
        if not matching:
            return
        key = matching[-1]
        record = self._records[key]
        updates: dict[str, str] = {}
        if chrome is not None:
            prior_chrome = record.chrome_delivery_result
            chrome_outcomes = {prior_chrome, chrome}
            updates["chrome_delivery_result"] = (
                "partial"
                if "delivered" in chrome_outcomes and "failed" in chrome_outcomes
                else chrome
            )
        if ntfy is not None:
            updates["ntfy_delivery_result"] = ntfy
        chrome_value = updates.get("chrome_delivery_result", record.chrome_delivery_result)
        ntfy_value = ntfy if ntfy is not None else record.ntfy_delivery_result
        delivery_values = {chrome_value, ntfy_value}
        failed = any(value in {"failed", "partial"} for value in delivery_values)
        delivered = "delivered" in delivery_values
        if failed and delivered:
            updates["notification_status"] = "partial_delivery"
        elif failed:
            updates["notification_status"] = "delivery_failed"
        elif delivered:
            updates["notification_status"] = "delivered"
        elif chrome_value == "not_connected":
            updates["notification_status"] = "delivery_failed"
        self._records[key] = record.model_copy(update=updates)

    def acknowledge(
        self,
        event_id: str,
        acknowledgement: OperatorAcknowledgement,
        acknowledged_at: datetime,
    ) -> SessionEventRecord | None:
        matching = [key for key in self._records if key[0] == event_id]
        if not matching:
            return None
        key = matching[-1]
        updated = self._records[key].model_copy(
            update={
                "operator_acknowledgement": acknowledgement,
                "acknowledged_at": acknowledged_at,
            }
        )
        self._records[key] = updated
        return updated
