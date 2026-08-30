import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.config import AppConfig, CircleGeofenceConfig, DetectionConfig
from mry_alert.detection.engine import DetectionEngine, DetectionEvent
from mry_alert.models import (
    AlertEvent,
    AlertEventType,
    ConfirmationStatus,
    DetectionDecision,
    NearbyAircraft,
    PendingDestinationEvent,
    SpeakerRole,
    TranscriptEvent,
)


def event(text: str, when: datetime) -> TranscriptEvent:
    return TranscriptEvent(event_id=str(uuid4()), timestamp=when, text=text)


def aircraft(registration: str = "N123AB") -> NearbyAircraft:
    return NearbyAircraft(
        hex=registration,
        registration=registration,
        flight=registration,
        aircraft_type="Citation 525",
        aircraft_type_code="C525",
        aircraft_type_name="Cessna Citation 525",
        aircraft_type_source="adsb_provider",
        aircraft_type_confidence=1.0,
        on_ground=True,
        ground_speed=7,
        seconds_since_seen=2,
    )


class SequenceProvider:
    def __init__(self, responses: list[list[NearbyAircraft]]) -> None:
        self.responses = responses
        self.calls = 0

    async def nearby(self) -> list[NearbyAircraft]:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def live_event(text: str, when: datetime) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=str(uuid4()),
        timestamp=when,
        text=text,
        source="liveatc_kmry_web_audio",
    )


def fast_config(delay: float = 0.02) -> AppConfig:
    return AppConfig(
        detection=DetectionConfig(
            destination_confirmation_delay_seconds=delay,
            destination_correction_window_seconds=20,
        )
    )


def make_engine(
    nearby: list[NearbyAircraft], delay: float = 0.02
) -> tuple[DetectionEngine, list[DetectionEvent]]:
    published: list[DetectionEvent] = []

    async def publish(destination_event: DetectionEvent) -> None:
        published.append(destination_event)

    return (
        DetectionEngine(fast_config(delay), MockNearbyAircraftProvider(nearby), publish),
        published,
    )


@pytest.mark.asyncio
async def test_a_no_correction_confirms_exactly_one_alert() -> None:
    engine, published = make_engine([aircraft()])
    now = datetime.now(UTC)
    pending = await engine.process(
        event(
            "Monterey Ground, November one two three alpha bravo, request taxi to "
            "Monterey Jet Center.",
            now,
        )
    )
    assert isinstance(pending, PendingDestinationEvent)
    assert pending.confirmation_status == ConfirmationStatus.PENDING
    assert pending.speaker_role == SpeakerRole.PILOT
    assert "first-person taxi request" in pending.speaker_role_reasons
    assert (
        await engine.process(
            event(
                "November one two three alpha bravo taxi to Monterey Jet Center",
                now + timedelta(seconds=1),
            )
        )
        is None
    )
    await asyncio.sleep(0.04)
    assert len([item for item in published if isinstance(item, PendingDestinationEvent)]) == 1
    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration == "N123AB"
    assert alerts[0].aircraft_type_name == "Cessna Citation 525"
    assert alerts[0].confirmation_status == ConfirmationStatus.CONFIRMED
    await engine.close()


@pytest.mark.asyncio
async def test_temporary_adsb_disappearance_preserves_valid_arrival_identity() -> None:
    published: list[DetectionEvent] = []

    async def publish(item: DetectionEvent) -> None:
        published.append(item)

    config = fast_config(0.01)
    provider = SequenceProvider([[aircraft()], []])
    engine = DetectionEngine(config, provider, publish)

    pending = await engine.process(
        live_event(
            "November one two three alpha bravo request taxi to Monterey Jet Center",
            datetime.now(UTC),
        )
    )
    assert isinstance(pending, PendingDestinationEvent)
    await asyncio.sleep(0.03)

    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration == "N123AB"
    await engine.close()


@pytest.mark.asyncio
async def test_later_adsb_observation_resolves_anonymous_arrival() -> None:
    published: list[DetectionEvent] = []

    async def publish(item: DetectionEvent) -> None:
        published.append(item)

    config = fast_config(0.01)
    config.adsb_decision.correlation_window_seconds = 0.06
    config.adsb.polling_interval_seconds = 0.01
    engine = DetectionEngine(config, SequenceProvider([[], [aircraft()]]), publish)

    pending = await engine.process(
        live_event(
            "We would like to go to Monterey Jet Center",
            datetime.now(UTC),
        )
    )
    assert isinstance(pending, PendingDestinationEvent)
    assert pending.registration is None
    await asyncio.sleep(0.04)

    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration == "N123AB"
    await engine.close()


@pytest.mark.asyncio
async def test_b_correction_before_delay_cancels_original_alert() -> None:
    engine, published = make_engine([aircraft()], delay=0.04)
    now = datetime.now(UTC)
    await engine.process(
        event("November one two three alpha bravo request taxi to Monterey Jet Center", now)
    )
    corrected = await engine.process(
        event(
            "November one two three alpha bravo wait, no, Del Monte Aviation",
            now + timedelta(seconds=5),
        )
    )
    await asyncio.sleep(0.06)
    assert isinstance(corrected, PendingDestinationEvent)
    assert corrected.confirmation_status == ConfirmationStatus.CORRECTED
    assert corrected.corrected_destination == "Del Monte Aviation"
    assert not any(isinstance(item, AlertEvent) for item in published)
    await engine.close()


@pytest.mark.asyncio
async def test_c_correction_after_publish_references_original_alert() -> None:
    engine, published = make_engine([aircraft()])
    now = datetime.now(UTC)
    await engine.process(
        event("November one two three alpha bravo request taxi to Monterey Jet Center", now)
    )
    await asyncio.sleep(0.04)
    original = next(item for item in published if isinstance(item, AlertEvent))
    correction = await engine.process(
        event(
            "November one two three alpha bravo actually parking at Del Monte Aviation instead",
            now + timedelta(seconds=5),
        )
    )
    assert isinstance(correction, AlertEvent)
    assert correction.event_type == AlertEventType.DESTINATION_CORRECTION
    assert correction.original_event_id == original.event_id
    assert correction.previous_destination == "Monterey Jet Center"
    assert correction.corrected_destination == "Del Monte Aviation"
    assert correction.aircraft_type_name == "Cessna Citation 525"
    await engine.close()


@pytest.mark.asyncio
async def test_d_other_aircraft_correction_does_not_cancel_pending_aircraft() -> None:
    engine, published = make_engine([aircraft(), aircraft("N456DE")], delay=0.04)
    now = datetime.now(UTC)
    await engine.process(
        event("November one two three alpha bravo request taxi to Monterey Jet Center", now)
    )
    await engine.process(
        event("November four five six delta echo, actually parking at Del Monte Aviation", now)
    )
    await asyncio.sleep(0.06)
    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration == "N123AB"
    await engine.close()


@pytest.mark.asyncio
async def test_e_unrelated_bare_no_does_not_cancel_pending_event() -> None:
    engine, published = make_engine([aircraft()])
    now = datetime.now(UTC)
    await engine.process(
        event("November one two three alpha bravo request taxi to Monterey Jet Center", now)
    )
    assert await engine.process(event("no reported traffic on the ramp", now)) is None
    await asyncio.sleep(0.04)
    assert len([item for item in published if isinstance(item, AlertEvent)]) == 1
    await engine.close()


@pytest.mark.asyncio
async def test_noisy_followup_does_not_erase_strong_pending_event() -> None:
    engine, published = make_engine([aircraft()], delay=0.02)
    now = datetime.now(UTC)
    pending = await engine.process(
        live_event(
            "November one two three alpha bravo request taxi to Monterey Jet Center",
            now,
        )
    )
    assert isinstance(pending, PendingDestinationEvent)

    noisy = live_event("unintelligible static broken transmission", now + timedelta(seconds=1))
    assert await engine.process(noisy) is None
    await asyncio.sleep(0.04)

    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration == "N123AB"
    await engine.close()


@pytest.mark.asyncio
async def test_mention_only_mode_does_not_turn_a_bare_mention_into_arrival() -> None:
    candidate = aircraft()
    candidate.latitude = 36.587
    candidate.longitude = -121.843
    config = fast_config()
    config.detection.alert_on_any_destination_mention = True
    engine = DetectionEngine(
        config,
        MockNearbyAircraftProvider([candidate]),
    )

    result = await engine.process(
        live_event("November one two three alpha bravo, Monterey Jet Center.", datetime.now(UTC))
    )

    assert result is None
    assert engine.config.detection.alert_on_any_destination_mention
    await engine.close()


@pytest.mark.asyncio
async def test_explicit_pilot_arrival_can_notify_clear_ground_match_immediately() -> None:
    candidate = aircraft()
    candidate.latitude = 36.587
    candidate.longitude = -121.843
    config = fast_config(delay=30)
    config.detection.alert_on_any_destination_mention = True
    config.detection.immediate_notification_on_clear_ground_match = True
    engine = DetectionEngine(config, MockNearbyAircraftProvider([candidate]))

    result = await engine.process(
        live_event(
            "November one two three alpha bravo, request taxi to Jet Center.",
            datetime.now(UTC),
        )
    )

    assert isinstance(result, AlertEvent)
    assert result.confirmation_status == ConfirmationStatus.CONFIRMED
    assert result.registration == "N123AB"
    await engine.close()


@pytest.mark.asyncio
async def test_mention_only_mode_rejects_airborne_candidate() -> None:
    candidate = aircraft()
    candidate.latitude = 36.587
    candidate.longitude = -121.843
    candidate.on_ground = False
    candidate.altitude = 1500
    candidate.ground_speed = 110
    config = fast_config()
    config.detection.alert_on_any_destination_mention = True
    engine = DetectionEngine(config, MockNearbyAircraftProvider([candidate]))

    result = await engine.process(
        live_event("November one two three alpha bravo, Jet Center.", datetime.now(UTC))
    )

    assert result is None
    await engine.close()


@pytest.mark.asyncio
async def test_mention_only_mode_does_not_guess_between_equal_aircraft() -> None:
    first = aircraft()
    first.latitude = 36.587
    first.longitude = -121.843
    second = aircraft("N456DE")
    second.latitude = 36.587
    second.longitude = -121.843
    config = fast_config()
    config.detection.alert_on_any_destination_mention = True
    engine = DetectionEngine(config, MockNearbyAircraftProvider([first, second]))

    result = await engine.process(live_event("Monterey Jet Center.", datetime.now(UTC)))

    assert result is None
    await engine.close()


@pytest.mark.asyncio
async def test_logged_weak_callsign_does_not_assign_closer_parked_tail() -> None:
    parked = aircraft("N100J")
    parked.hex = "parked"
    parked.latitude = 36.586
    parked.longitude = -121.842
    parked.ground_speed = 0
    actual = aircraft("N124CK")
    actual.hex = "actual"
    actual.latitude = 36.59
    actual.longitude = -121.843
    actual.ground_speed = 0
    config = fast_config(delay=30)
    config.adsb_geofences.monterey_jet_center = CircleGeofenceConfig(
        latitude=36.586,
        longitude=-121.842,
        radius_nm=0.2,
    )
    config.detection.alert_on_any_destination_mention = True
    config.detection.immediate_notification_on_clear_ground_match = True
    engine = DetectionEngine(
        config,
        MockNearbyAircraftProvider([parked, actual]),
    )
    first = live_event(
        "Monterey Ground, number one for Charlie Kilo. Clear off runway Two Eight "
        "Left. We would like to go to the Monterey Jet Center.",
        datetime.now(UTC),
    )

    anonymous = await engine.process(first)

    assert isinstance(anonymous, PendingDestinationEvent)
    assert anonymous.registration is None
    assert first.detection_decision == DetectionDecision.PENDING
    assert first.identified_registration is None

    clarified = await engine.process(
        live_event(
            "Two Four Charlie Kilo, request taxi to Jet Center via Alpha and Echo.",
            first.timestamp + timedelta(seconds=5),
        )
    )

    assert isinstance(clarified, AlertEvent)
    assert clarified.registration == "N124CK"
    assert "unique suffix matched nearby registration" in clarified.match_reasons
    await engine.close()


@pytest.mark.asyncio
async def test_mention_only_mode_is_disabled_by_default() -> None:
    candidate = aircraft()
    candidate.latitude = 36.587
    candidate.longitude = -121.843
    engine = DetectionEngine(fast_config(), MockNearbyAircraftProvider([candidate]))

    result = await engine.process(live_event("Monterey Jet Center.", datetime.now(UTC)))

    assert result is None
    await engine.close()


@pytest.mark.asyncio
async def test_split_exchange_uses_context_and_suffix() -> None:
    engine, published = make_engine([aircraft()])
    now = datetime.now(UTC)
    assert await engine.process(event("Citation three alpha bravo, say parking.", now)) is None
    pending = await engine.process(event("Monterey Jet Center.", now + timedelta(seconds=10)))
    assert isinstance(pending, PendingDestinationEvent)
    assert pending.speaker_role == SpeakerRole.PILOT
    assert "short destination reply" in " ".join(pending.speaker_role_reasons)
    await asyncio.sleep(0.04)
    alert = next(item for item in published if isinstance(item, AlertEvent))
    assert alert.registration == "N123AB"
    assert "unique suffix matched nearby registration" in alert.match_reasons
    await engine.close()


@pytest.mark.asyncio
async def test_controller_instruction_mentioning_mjc_does_not_create_event() -> None:
    engine, published = make_engine([aircraft()])
    transmission = event(
        "November one two three alpha bravo taxi to Monterey Jet Center",
        datetime.now(UTC),
    )
    result = await engine.process(transmission)
    assert isinstance(result, PendingDestinationEvent)
    assert transmission.speaker_role == SpeakerRole.CONTROLLER
    assert transmission.speaker_role_confidence >= 0.8
    await asyncio.sleep(0.04)
    assert not any(isinstance(item, AlertEvent) for item in published)
    await engine.close()


@pytest.mark.asyncio
async def test_interleaved_contact_blocks_ambiguous_short_destination_reply() -> None:
    engine, published = make_engine([aircraft(), aircraft("N456DE")])
    now = datetime.now(UTC)
    await engine.process(event("Citation three alpha bravo, say parking", now))
    await engine.process(
        event("November four five six delta echo ready to taxi", now + timedelta(seconds=1))
    )
    short_reply = event("Monterey Jet Center", now + timedelta(seconds=2))
    assert await engine.process(short_reply) is None
    assert short_reply.speaker_role == SpeakerRole.UNKNOWN
    assert published == []
    await engine.close()


@pytest.mark.asyncio
async def test_irrelevant_and_bare_destination_create_no_event() -> None:
    engine, _ = make_engine([aircraft()])
    now = datetime.now(UTC)
    assert await engine.process(event("Monterey Tower, N456DE ready for departure", now)) is None
    assert await engine.process(event("Monterey Jet Center", now)) is None
    await engine.close()


@pytest.mark.asyncio
async def test_adsb_failure_does_not_crash_pipeline() -> None:
    engine = DetectionEngine(
        fast_config(), MockNearbyAircraftProvider(error=TimeoutError("offline"))
    )
    result = await engine.process(
        live_event(
            "November one two three alpha bravo request taxi to Monterey Jet Center",
            datetime.now(UTC),
        )
    )
    assert isinstance(result, PendingDestinationEvent)
    await asyncio.sleep(0.04)
    await engine.close()


@pytest.mark.asyncio
async def test_callsign_only_followup_resolves_anonymous_arrival_without_guessing() -> None:
    published: list[DetectionEvent] = []

    async def publish(item: DetectionEvent) -> None:
        published.append(item)

    first = aircraft("N111AA")
    second = aircraft("N222BB")
    config = fast_config(delay=10)
    config.adsb_decision.correlation_window_seconds = 5
    engine = DetectionEngine(config, MockNearbyAircraftProvider([first, second]), publish)
    now = datetime.now(UTC)

    pending = await engine.process(
        live_event("we would like to go to Monterey Jet Center", now)
    )
    assert isinstance(pending, PendingDestinationEvent)
    assert pending.registration is None

    resolved = await engine.process(
        live_event("november two two two bravo bravo", now + timedelta(seconds=2))
    )

    assert isinstance(resolved, AlertEvent)
    assert resolved.registration == "N222BB"
    assert "later callsign/readback linked" in " ".join(resolved.match_reasons)
    await engine.close()


@pytest.mark.asyncio
async def test_repeated_spoken_registration_can_alert_without_wrong_adsb_tail() -> None:
    config = fast_config(delay=10)
    config.adsb_decision.correlation_window_seconds = 30
    candidates = [aircraft("N74XP"), aircraft("N964QS")]
    engine = DetectionEngine(config, MockNearbyAircraftProvider(candidates))
    now = datetime.now(UTC)

    pending = await engine.process(
        live_event(
            "november four six five charlie golf we d like to taxi to Monterey Jet Center",
            now,
        )
    )
    assert isinstance(pending, PendingDestinationEvent)

    resolved = await engine.process(
        live_event(
            "november four six five charlie golf taxi to Jet Center via Alpha Echo",
            now + timedelta(seconds=12),
        )
    )

    assert isinstance(resolved, AlertEvent)
    assert resolved.registration == "N465CG"
    assert resolved.status.value == "likely"
    assert "unrelated ADS-B tail was not guessed" in resolved.match_reasons
    await engine.close()


@pytest.mark.asyncio
async def test_strong_pilot_arrival_eventually_alerts_unidentified(
) -> None:
    published: list[DetectionEvent] = []

    async def publish(item: DetectionEvent) -> None:
        published.append(item)

    config = fast_config(delay=0.01)
    config.adsb_decision.correlation_window_seconds = 0.04
    config.adsb.polling_interval_seconds = 0.01
    engine = DetectionEngine(
        config,
        MockNearbyAircraftProvider([aircraft("N111AA"), aircraft("N222BB")]),
        publish,
    )

    pending = await engine.process(
        live_event("we would like to go to Monterey Jet Center", datetime.now(UTC))
    )
    assert isinstance(pending, PendingDestinationEvent)
    await asyncio.sleep(0.08)

    alerts = [item for item in published if isinstance(item, AlertEvent)]
    assert len(alerts) == 1
    assert alerts[0].registration is None
    assert "without guessing an aircraft tail" in " ".join(alerts[0].match_reasons)
    await engine.close()
