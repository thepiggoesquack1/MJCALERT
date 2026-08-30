import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.config import AppConfig, DetectionConfig
from mry_alert.detection.artifacts import trim_repetitive_tail
from mry_alert.detection.engine import DetectionEngine, DetectionEvent
from mry_alert.detection.intent_normalizer import normalize_ground_intent
from mry_alert.detection.normalizer import normalize_transcript
from mry_alert.models import (
    AlertEvent,
    DetectionDecision,
    NearbyAircraft,
    PendingDestinationEvent,
    TranscriptEvent,
)


def live_event(text: str) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=str(uuid4()),
        timestamp=datetime.now(UTC),
        text=text,
        source="liveatc_kmry_web_audio",
    )


def ground_aircraft(registration: str = "N825SP") -> NearbyAircraft:
    return NearbyAircraft(
        hex=registration,
        registration=registration,
        on_ground=True,
        ground_speed=8,
        seconds_since_seen=2,
        distance_nm=1,
    )


def engine_with(aircraft: list[NearbyAircraft]) -> tuple[DetectionEngine, list[DetectionEvent]]:
    published: list[DetectionEvent] = []

    async def publish(event: DetectionEvent) -> None:
        published.append(event)

    config = AppConfig(detection=DetectionConfig(destination_confirmation_delay_seconds=0.01))
    return DetectionEngine(config, MockNearbyAircraftProvider(aircraft), publish), published


def test_parking_prompt_recovery_is_source_and_context_scoped() -> None:
    normalized = normalize_transcript("Citation three alpha bravo, say barking")
    recovered = normalize_ground_intent(normalized, authorized_ground_source=True)
    assert "say parking" in recovered.text
    assert "ATC parking-prompt pattern" in recovered.reasons[0]
    unrelated = normalize_ground_intent(
        normalize_transcript("The dog is barking near the hangar"),
        authorized_ground_source=True,
    )
    assert unrelated.text == "the dog is barking near the hangar"
    assert not unrelated.reasons
    outside_authorized_source = normalize_ground_intent(
        normalize_transcript("say barking"), authorized_ground_source=False
    )
    assert outside_authorized_source.text == "say barking"


@pytest.mark.asyncio
async def test_recovered_parking_prompt_links_short_mjc_reply() -> None:
    engine, _ = engine_with([ground_aircraft("N123AB")])
    assert await engine.process(live_event("Citation three alpha bravo, say barking")) is None
    result = await engine.process(live_event("Monterey Jet Center"))
    assert isinstance(result, PendingDestinationEvent)
    assert result.registration == "N123AB"
    await engine.close()


def test_repetitive_decoder_tail_is_trimmed_without_removing_go_around() -> None:
    result = trim_repetitive_tail("Alpha Echo to the Jet Center " + "go " * 22)
    assert result.text == "Alpha Echo to the Jet Center"
    assert result.reason and "removed 22" in result.reason
    assert trim_repetitive_tail("go around").text == "go around"


@pytest.mark.asyncio
async def test_route_clearance_without_callsign_creates_pending_likely_event() -> None:
    engine, _ = engine_with([ground_aircraft()])
    result = await engine.process(live_event("Alpha Echo to the Jet Center"))
    assert isinstance(result, PendingDestinationEvent)
    assert result.registration == "N825SP"
    assert result.identification_source.value == "adsb_correlation"
    await engine.close()


@pytest.mark.asyncio
async def test_observed_garbled_route_regression_creates_pending() -> None:
    engine, published = engine_with([])
    result = await engine.process(
        live_event(
            "Good to hit 9.50, continue down the runway, turn left at Foxtrot, "
            "say barking. Foxtrot was going on Monterey Jet."
        )
    )
    assert isinstance(result, PendingDestinationEvent)
    await asyncio.sleep(0.03)
    assert not any(isinstance(item, AlertEvent) for item in published)
    await engine.close()


@pytest.mark.asyncio
async def test_generic_business_mention_remains_ignored() -> None:
    engine, published = engine_with([ground_aircraft()])
    result = await engine.process(live_event("Call Monterey Jet Center later"))
    assert result is None
    await asyncio.sleep(0.02)
    assert not any(isinstance(item, AlertEvent) for item in published)
    await engine.close()


@pytest.mark.asyncio
async def test_ground_route_fallback_is_not_enabled_for_other_sources() -> None:
    engine, _ = engine_with([ground_aircraft()])
    transmission = TranscriptEvent(
        event_id=str(uuid4()),
        timestamp=datetime.now(UTC),
        text="Alpha Echo to the Jet Center",
        source="simulation",
    )
    assert await engine.process(transmission) is None
    await engine.close()


@pytest.mark.asyncio
async def test_route_cues_split_across_linked_transmissions_are_preserved() -> None:
    engine, _ = engine_with([ground_aircraft("N123AB")])
    await engine.process(live_event("November one two three alpha bravo taxi via Alpha Echo"))

    result = await engine.process(live_event("November one two three alpha bravo Jet Center"))

    assert isinstance(result, PendingDestinationEvent)
    assert result.registration == "N123AB"
    assert "recent linked route context" in " ".join(result.match_reasons)
    await engine.close()


@pytest.mark.asyncio
async def test_weak_phrase_without_route_is_unresolved_not_ignored() -> None:
    engine, _ = engine_with([ground_aircraft("N123AB")])
    transmission = live_event("November one two three alpha bravo Jet Center")

    assert await engine.process(transmission) is None
    assert transmission.detection_decision == DetectionDecision.UNRESOLVED
    await engine.close()


@pytest.mark.asyncio
async def test_adsb_alone_cannot_create_jet_center_intent() -> None:
    engine, published = engine_with([ground_aircraft("N123AB")])
    transmission = live_event("November one two three alpha bravo holding short runway two eight")

    assert await engine.process(transmission) is None
    assert transmission.destination_candidate is None
    assert published == []
    await engine.close()
