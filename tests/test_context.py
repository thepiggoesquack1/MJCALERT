from datetime import UTC, datetime, timedelta

from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.context import ConversationContext
from mry_alert.models import SpeakerRole


def test_unique_parking_contact_is_inherited() -> None:
    now = datetime.now(UTC)
    context = ConversationContext(120)
    callsign = parse_callsign("citation three alpha bravo")
    assert callsign
    context.observe(
        callsign,
        "citation three alpha bravo say parking",
        now,
        SpeakerRole.CONTROLLER,
        0.95,
    )
    inherited, ambiguous = context.parking_response_contact(now + timedelta(seconds=10))
    assert inherited == callsign
    assert not ambiguous


def test_multiple_waiting_contacts_are_ambiguous() -> None:
    now = datetime.now(UTC)
    context = ConversationContext(120)
    first = parse_callsign("citation three alpha bravo")
    second = parse_callsign("citation four delta echo")
    assert first and second
    context.observe(first, "say parking", now, SpeakerRole.CONTROLLER, 0.95)
    context.observe(second, "say parking", now, SpeakerRole.CONTROLLER, 0.95)
    inherited, ambiguous = context.parking_response_contact(now + timedelta(seconds=1))
    assert inherited is None and ambiguous


def test_newer_competing_contact_makes_short_reply_ambiguous() -> None:
    now = datetime.now(UTC)
    context = ConversationContext(120)
    first = parse_callsign("citation three alpha bravo")
    second = parse_callsign("citation four delta echo")
    assert first and second
    context.observe(first, "say parking", now, SpeakerRole.CONTROLLER, 0.95)
    context.observe(second, "ready to taxi", now + timedelta(seconds=1), SpeakerRole.PILOT, 0.9)
    inherited, ambiguous = context.parking_response_contact(now + timedelta(seconds=2))
    assert inherited is None and ambiguous
