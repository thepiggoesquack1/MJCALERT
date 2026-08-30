from __future__ import annotations

from datetime import UTC, datetime

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.config import AppConfig
from mry_alert.detection.engine import DetectionEngine
from mry_alert.detection.outbound import (
    OutboundDisposition,
    infer_outbound_departure,
)
from mry_alert.models import (
    AdsbDecisionState,
    AircraftMatch,
    AlertEvent,
    DetectionDecision,
    IdentificationSource,
    MatchStatus,
    NearbyAircraft,
    PendingDestinationEvent,
    TranscriptEvent,
)


def movement_match(**updates: object) -> AircraftMatch:
    aircraft = NearbyAircraft(
        hex="abc123",
        registration="N123AB",
        aircraft_type="C525",
        latitude=36.589,
        longitude=-121.858,
        on_ground=True,
        ground_speed=4,
        seconds_since_seen=1,
    )
    values: dict[str, object] = {
        "aircraft": aircraft,
        "registration": "N123AB",
        "confidence": 0.92,
        "status": MatchStatus.CONFIRMED,
        "match_reasons": ["one ADS-B candidate clearly won"],
        "candidate_scores": ["N123AB score=100.0 state=taxiing: on_ground"],
        "identification_source": IdentificationSource.ADSB_CORRELATION,
        "adsb_decision": AdsbDecisionState.CONFIRMED,
        "inside_fbo_geofence": True,
        "fbo_geofence_names": ["Monterey Jet Center"],
        "was_stationary_at_fbo": True,
        "movement_state": "taxiing",
    }
    values.update(updates)
    return AircraftMatch.model_validate(values)


def transcript(text: str) -> TranscriptEvent:
    return TranscriptEvent(
        event_id=text,
        timestamp=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        text=text,
        source="liveatc_kmry_web_audio",
    )


def engine_with_match(match: AircraftMatch) -> DetectionEngine:
    config = AppConfig()
    config.detection.destination_confirmation_delay_seconds = 60
    engine = DetectionEngine(config, MockNearbyAircraftProvider([]))
    engine.matcher.match = lambda *_args, **_kwargs: match  # type: ignore[method-assign]
    return engine


async def test_arrival_to_jet_center_still_creates_pending_alert() -> None:
    engine = engine_with_match(
        movement_match(
            inside_fbo_geofence=False,
            fbo_geofence_names=[],
            was_stationary_at_fbo=False,
            movement_state="moving_toward_jet_center",
        )
    )
    event = transcript(
        "November one two three alpha bravo request taxi via Alpha Echo to Monterey Jet Center"
    )

    result = await engine.process(event)
    await engine.close()

    assert isinstance(result, PendingDestinationEvent)
    assert event.detection_decision == DetectionDecision.PENDING


async def test_outbound_parking_report_is_ignored() -> None:
    engine = engine_with_match(movement_match(movement_state="moving_away_from_jet_center"))
    event = transcript(
        "November one two three alpha bravo at Monterey Jet Center "
        "request taxi for departure runway two eight"
    )

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.IGNORED
    assert "current location" in " ".join(event.detection_reasons)


async def test_from_jet_center_is_filtered_as_outbound() -> None:
    engine = engine_with_match(movement_match(movement_state="moving_away_from_jet_center"))
    event = transcript("November one two three alpha bravo from the Jet Center request taxi")

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.IGNORED


async def test_current_location_without_departure_evidence_is_unresolved() -> None:
    engine = engine_with_match(
        movement_match(was_stationary_at_fbo=False, movement_state="taxiing")
    )
    event = transcript("November one two three alpha bravo we are at Monterey Jet Center")

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.UNRESOLVED


async def test_taxi_via_inside_jet_center_is_filtered_as_outbound() -> None:
    engine = engine_with_match(movement_match(movement_state="parked_at_jet_center"))
    event = transcript(
        "November one two three alpha bravo at Monterey Jet Center taxi via Alpha Echo"
    )

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.IGNORED


async def test_controller_parking_prompt_does_not_create_arrival() -> None:
    engine = engine_with_match(movement_match())
    event = transcript("November one two three alpha bravo say parking")

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.IGNORED


async def test_conservative_mode_keeps_controller_route_pending() -> None:
    engine = engine_with_match(
        movement_match(
            inside_fbo_geofence=False,
            was_stationary_at_fbo=False,
            movement_state="moving_toward_jet_center",
        )
    )
    event = transcript("November one two three alpha bravo taxi Jet Center via Alpha Echo")

    result = await engine.process(event)
    await engine.close()

    assert isinstance(result, PendingDestinationEvent)
    assert event.detection_decision == DetectionDecision.PENDING


async def test_balanced_mode_alerts_controller_route_with_strong_toward_adsb() -> None:
    engine = engine_with_match(
        movement_match(
            inside_fbo_geofence=False,
            was_stationary_at_fbo=False,
            movement_state="moving_toward_jet_center",
        )
    )
    engine.config.detection.alert_sensitivity = "balanced"
    event = transcript("November one two three alpha bravo taxi Jet Center via Alpha Echo")

    result = await engine.process(event)
    await engine.close()

    assert isinstance(result, AlertEvent)
    assert event.detection_decision == DetectionDecision.CONFIRMED
    assert result.status == MatchStatus.LIKELY
    assert "balanced mode accepted controller routing" in " ".join(result.match_reasons)


async def test_never_miss_mode_alerts_exact_mention_with_plausible_ground_adsb() -> None:
    engine = engine_with_match(
        movement_match(
            inside_fbo_geofence=False,
            was_stationary_at_fbo=False,
            movement_state="taxiing",
        )
    )
    engine.config.detection.alert_sensitivity = "never_miss"
    event = transcript("Monterey Jet Center")

    result = await engine.process(event)
    await engine.close()

    assert isinstance(result, AlertEvent)
    assert event.detection_decision == DetectionDecision.CONFIRMED
    assert result.status == MatchStatus.LIKELY
    assert "never-miss mode accepted" in " ".join(result.match_reasons)


async def test_never_miss_mode_preserves_outbound_filter() -> None:
    engine = engine_with_match(movement_match(movement_state="moving_away_from_jet_center"))
    engine.config.detection.alert_sensitivity = "never_miss"
    event = transcript(
        "November one two three alpha bravo at Monterey Jet Center request taxi for departure"
    )

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.IGNORED


async def test_ambiguous_parking_movement_remains_unresolved() -> None:
    engine = engine_with_match(
        movement_match(was_stationary_at_fbo=False, movement_state="taxiing")
    )
    event = transcript("November one two three alpha bravo parking at Monterey Jet Center")

    result = await engine.process(event)
    await engine.close()

    assert result is None
    assert event.detection_decision == DetectionDecision.UNRESOLVED
    assert "not strong enough to suppress" in " ".join(event.detection_reasons)


def test_explicit_inbound_wording_wins_over_current_position() -> None:
    decision = infer_outbound_departure(
        "taxi to the Monterey Jet Center",
        movement_match(movement_state="moving_away_from_jet_center"),
    )

    assert decision.disposition == OutboundDisposition.ARRIVAL_ELIGIBLE
