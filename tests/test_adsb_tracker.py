from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.adsb.tracker import AdsbCorrelator, MovementState, haversine_nm
from mry_alert.cli import _load_adsb_fixture
from mry_alert.config import AppConfig, CircleGeofenceConfig
from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.engine import DetectionEngine
from mry_alert.models import (
    AdsbDecisionState,
    NearbyAircraft,
    PendingDestinationEvent,
    SpokenCallsign,
    TranscriptEvent,
)


def correlator() -> AdsbCorrelator:
    config = AppConfig()
    config.adsb_geofences.monterey_jet_center = CircleGeofenceConfig(
        latitude=36.586, longitude=-121.842, radius_nm=0.2
    )
    return AdsbCorrelator(
        config.airport,
        config.adsb_tracking,
        config.adsb_geofences,
        config.adsb_matching,
        config.adsb_scoring,
        config.adsb_decision,
    )


def aircraft(
    registration: str = "N825SP",
    *,
    hex_value: str = "abc",
    latitude: float | None = 36.587,
    longitude: float | None = -121.843,
    speed: float | None = 8,
    on_ground: bool = True,
    altitude: float | None = 100,
    seen: float = 1,
) -> NearbyAircraft:
    return NearbyAircraft(
        hex=hex_value,
        registration=registration,
        latitude=latitude,
        longitude=longitude,
        ground_speed=speed,
        on_ground=on_ground,
        altitude=altitude,
        seconds_since_seen=seen,
    )


def spoken(registration: str = "N825SP") -> SpokenCallsign:
    return SpokenCallsign(
        original_text=registration,
        normalized_form=registration,
        full_registration=registration,
        parse_confidence=0.98,
    )


def test_haversine_and_moving_toward_trend() -> None:
    assert haversine_nm(36.587, -121.843, 36.587, -121.843) == pytest.approx(0)
    now = datetime.now(UTC)
    far = aircraft(latitude=36.59, longitude=-121.85)
    far.source_timestamp = now - timedelta(seconds=5)
    near = aircraft(latitude=36.587, longitude=-121.843)
    near.source_timestamp = now
    match = correlator().correlate(None, [far, near], now)
    assert any("moving_toward_jet_center" in line for line in match.candidate_scores)


def test_moving_away_trend_is_negative_evidence() -> None:
    now = datetime.now(UTC)
    near = aircraft(latitude=36.587, longitude=-121.843)
    near.source_timestamp = now - timedelta(seconds=5)
    far = aircraft(latitude=36.59, longitude=-121.85)
    far.source_timestamp = now
    match = correlator().correlate(None, [near, far], now)
    assert any("moving_away" in line for line in match.candidate_scores)


def test_incomplete_latest_sample_does_not_change_movement_direction() -> None:
    now = datetime.now(UTC)
    tracker = correlator()
    far = aircraft(latitude=36.59, longitude=-121.85)
    far.source_timestamp = now - timedelta(seconds=10)
    near = aircraft(latitude=36.587, longitude=-121.843)
    near.source_timestamp = now - timedelta(seconds=5)
    tracker.correlate(None, [far, near], now - timedelta(seconds=5))
    incomplete = aircraft(latitude=None, longitude=None)
    incomplete.source_timestamp = now

    match = tracker.correlate(None, [incomplete], now)

    assert not any("moving_toward_jet_center" in line for line in match.candidate_scores)
    assert not any("moving_away" in line for line in match.candidate_scores)


def test_recently_landed_is_preferred() -> None:
    now = datetime.now(UTC)
    airborne = aircraft(on_ground=False, altitude=1000, speed=90)
    airborne.source_timestamp = now - timedelta(seconds=20)
    landed = aircraft(on_ground=True, altitude=0, speed=12)
    landed.source_timestamp = now
    match = correlator().correlate(None, [airborne, landed], now)
    assert any(MovementState.RECENTLY_LANDED.value in line for line in match.candidate_scores)


def test_exact_speech_and_strong_adsb_confirm() -> None:
    match = correlator().correlate(spoken(), [aircraft()])
    assert match.adsb_decision == AdsbDecisionState.CONFIRMED
    assert match.registration == "N825SP"


def test_no_spoken_callsign_can_confirm_one_strong_candidate() -> None:
    match = correlator().correlate(None, [aircraft()])
    assert match.adsb_decision == AdsbDecisionState.CONFIRMED


def test_speech_adsb_conflict_is_rejected() -> None:
    match = correlator().correlate(spoken("N731AB"), [aircraft()])
    assert match.adsb_decision == AdsbDecisionState.REJECTED
    assert match.registration is None
    assert "speech_adsb_conflict" in match.match_reasons


def test_close_candidates_are_ambiguous() -> None:
    match = correlator().correlate(
        None, [aircraft(hex_value="a"), aircraft("N731AB", hex_value="b")]
    )
    assert match.adsb_decision == AdsbDecisionState.AMBIGUOUS
    assert match.registration is None


def test_exact_match_exceeds_margin() -> None:
    match = correlator().correlate(
        spoken(), [aircraft(hex_value="a"), aircraft("N731AB", hex_value="b")]
    )
    assert match.adsb_decision == AdsbDecisionState.CONFIRMED
    assert match.winning_margin and match.winning_margin >= 15


def test_stale_candidate_rejected() -> None:
    match = correlator().correlate(None, [aircraft(seen=30)])
    assert match.adsb_decision == AdsbDecisionState.REJECTED
    assert "stale_candidate_rejected" in match.match_reasons


def test_airborne_climbing_candidate_rejected() -> None:
    candidate = aircraft(on_ground=False, altitude=1200, speed=100)
    candidate.vertical_rate = 800
    match = correlator().correlate(spoken(), [candidate])
    assert match.adsb_decision == AdsbDecisionState.REJECTED


@pytest.mark.parametrize(
    "updates",
    [
        {"altitude": None},
        {"ground_speed": None},
        {"on_ground": False, "altitude": None},
        {"latitude": None, "longitude": None},
    ],
)
def test_missing_adsb_fields_are_safe(updates: dict[str, object]) -> None:
    candidate = aircraft().model_copy(update=updates)
    result = correlator().correlate(None, [candidate])
    assert result.adsb_decision is not None


def test_duplicate_registration_is_penalized_and_ambiguous() -> None:
    match = correlator().correlate(None, [aircraft(hex_value="a"), aircraft(hex_value="b")])
    assert match.registration is None
    assert any("duplicate_registration" in line for line in match.candidate_scores)


def test_no_candidate_has_explicit_decision() -> None:
    assert correlator().correlate(None, []).adsb_decision == AdsbDecisionState.NO_CANDIDATE


def test_taxiing_candidate_scores_above_stopped_distant_candidate() -> None:
    moving = aircraft(hex_value="moving", speed=8)
    stopped = aircraft("N731AB", hex_value="stopped", latitude=36.60, speed=0)
    match = correlator().correlate(None, [moving, stopped])
    assert match.registration == "N825SP"


def test_parked_aircraft_cannot_be_assigned_without_speech_identity() -> None:
    parked = aircraft(
        "N100J",
        hex_value="parked",
        latitude=36.586,
        longitude=-121.842,
        speed=0,
    )
    stopped = aircraft(
        "N124CK",
        hex_value="actual",
        latitude=36.59,
        longitude=-121.843,
        speed=0,
    )

    match = correlator().correlate(None, [parked, stopped])

    assert match.registration is None
    assert match.adsb_decision == AdsbDecisionState.INSUFFICIENT_DATA
    assert "stationary ADS-B candidate" in " ".join(match.match_reasons)


def test_unique_spoken_suffix_beats_closer_unrelated_parked_aircraft() -> None:
    parked = aircraft(
        "N100J",
        hex_value="parked",
        latitude=36.586,
        longitude=-121.842,
        speed=0,
    )
    actual = aircraft(
        "N124CK",
        hex_value="actual",
        latitude=36.59,
        longitude=-121.843,
        speed=0,
    )
    callsign = parse_callsign("two four charlie kilo monterey ground")
    assert callsign is not None

    match = correlator().correlate(callsign, [parked, actual])

    assert match.registration == "N124CK"
    assert match.adsb_decision == AdsbDecisionState.CONFIRMED
    assert "unique suffix matched nearby registration" in match.match_reasons


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_uncertain", [False, True])
async def test_pilot_arrival_stays_pending_while_adsb_is_ambiguous(
    allow_uncertain: bool,
) -> None:
    config = AppConfig()
    config.notifications.send_uncertain_alerts = allow_uncertain
    config.detection.destination_confirmation_delay_seconds = 0
    candidates = [aircraft(hex_value="a"), aircraft("N731AB", hex_value="b")]
    engine = DetectionEngine(config, MockNearbyAircraftProvider(candidates))
    event = TranscriptEvent(
        event_id="route",
        timestamp=datetime.now(UTC),
        text="We request taxi via Alpha Echo to the Monterey Jet Center",
        source=config.liveatc.source_label,
    )
    result = await engine.process(event)
    assert isinstance(result, PendingDestinationEvent)
    await engine.close()


def test_adsb_fixture_loading(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        '{"observations":[{"timestamp_offset_seconds":0,"hex":"abc",'
        '"registration":"N825SP","altitude_ft":100,"ground_speed_knots":8}]}',
        encoding="utf-8",
    )
    loaded = _load_adsb_fixture(path)
    assert loaded[0].registration == "N825SP"
    assert loaded[0].ground_speed == 8
