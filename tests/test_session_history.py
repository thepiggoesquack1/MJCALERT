from __future__ import annotations

from datetime import UTC, datetime

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
from mry_alert.server.session_history import (
    SessionEventHistory,
    record_detection_event,
    record_transcript,
)

NOW = datetime(2026, 8, 5, 18, 0, tzinfo=UTC)


def transcript(event_id: str, reason: str) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=event_id,
        timestamp=NOW,
        text="November six two seven sierra at Monterey Jet Center request taxi",
        detected_callsign="N627S",
        destination_candidate="Monterey Jet Center",
        detection_decision=DetectionDecision.IGNORED,
        detection_reasons=[reason],
        identified_registration="N627S",
    )


def alert(event_id: str = "arrival-1") -> AlertEvent:
    return AlertEvent(
        event_id=event_id,
        timestamp=NOW,
        destination="Monterey Jet Center",
        registration="N627S",
        spoken_callsign="November six two seven sierra",
        aircraft_type="Cessna Citation CJ3",
        aircraft_type_code="C25B",
        aircraft_type_name="Cessna Citation CJ3",
        aircraft_type_source="adsb_provider",
        aircraft_type_confidence=1.0,
        confidence=0.92,
        status=MatchStatus.CONFIRMED,
        transcript_excerpt="We would like to go to Monterey Jet Center",
        match_reasons=["strong pilot arrival evidence"],
    )


def test_confirmed_denied_and_suppressed_outcomes_are_recorded() -> None:
    confirmed = record_detection_event(alert())
    denied = record_transcript(transcript("denied-1", "destination was not requested"))
    outbound = record_transcript(
        transcript("outbound-1", "outbound_departure_filtered")
    )

    assert confirmed.transition_type == "confirmed"
    assert confirmed.aircraft_type_code == "C25B"
    assert denied.transition_type == "denied"
    assert outbound.transition_type == "outbound_filtered"
    assert outbound.notification_status == "suppressed"


def test_correction_links_to_original_event() -> None:
    correction = alert("correction-1").model_copy(
        update={
            "event_type": AlertEventType.DESTINATION_CORRECTION,
            "original_event_id": "arrival-1",
            "corrected_destination": "Del Monte Aviation",
        }
    )
    record = record_detection_event(correction)

    assert record.transition_type == "corrected"
    assert record.original_event_id == "arrival-1"
    assert record.aircraft_type_name == "Cessna Citation CJ3"


def test_duplicate_records_are_replaced_and_buffer_is_bounded() -> None:
    history = SessionEventHistory(maximum=2)
    first = record_detection_event(alert("one"))
    history.append(first)
    history.append(first.model_copy(update={"notification_status": "updated"}))
    history.append(record_detection_event(alert("two")))
    history.append(record_detection_event(alert("three")))

    records = history.recent()
    assert [item.event_id for item in records] == ["three", "two"]


def test_delivery_updates_do_not_store_secrets() -> None:
    history = SessionEventHistory()
    history.append(record_detection_event(alert()))
    history.update_delivery("arrival-1", chrome="delivered", ntfy="failed")

    dumped = history.recent()[0].model_dump_json()
    assert '"chrome_delivery_result":"delivered"' in dumped
    assert '"ntfy_delivery_result":"failed"' in dumped
    assert "pairing" not in dumped.casefold()
    assert "authorization" not in dumped.casefold()


def test_session_history_has_no_persistence_api() -> None:
    history = SessionEventHistory()
    assert not hasattr(history, "path")
    assert not hasattr(history, "save")


def test_record_model_accepts_unknown_aircraft_type() -> None:
    record = SessionEventRecord(
        event_id="unknown",
        transition_type="possible",
        timestamp=NOW,
        final_decision="possible",
    )
    assert record.aircraft_type is None


def test_terminal_transition_replaces_pending_row() -> None:
    pending = PendingDestinationEvent(
        **alert().model_dump(exclude={"test", "confirmation_status"}),
        contact_key="november627s",
        confirmation_status=ConfirmationStatus.PENDING,
    )
    history = SessionEventHistory()
    history.append(record_detection_event(pending))
    history.append(record_detection_event(alert()))

    assert [(item.event_id, item.transition_type) for item in history.recent()] == [
        ("arrival-1", "confirmed")
    ]


def test_expired_pending_and_operator_acknowledgement_are_session_state() -> None:
    pending = PendingDestinationEvent(
        **alert().model_dump(exclude={"test", "confirmation_status"}),
        contact_key="november627s",
        confirmation_status=ConfirmationStatus.PENDING,
    )
    history = SessionEventHistory()
    history.append(record_detection_event(pending))
    history.append(
        record_detection_event(
            pending.model_copy(
                update={"confirmation_status": ConfirmationStatus.EXPIRED}
            )
        )
    )
    updated = history.acknowledge(
        "arrival-1", OperatorAcknowledgement.FALSE_DETECTION, NOW
    )

    assert updated is not None
    assert updated.transition_type == "expired"
    assert updated.operator_acknowledgement == OperatorAcknowledgement.FALSE_DETECTION
    assert updated.acknowledged_at == NOW
    assert len(history.recent()) == 1
