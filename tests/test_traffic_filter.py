from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.adsb.traffic_filter import (
    TrafficDecision,
    TrafficFilter,
    TrafficFilteringProvider,
)
from mry_alert.config import AppConfig, TrafficFilterConfig
from mry_alert.detection.engine import DetectionEngine
from mry_alert.models import NearbyAircraft, PendingDestinationEvent, TranscriptEvent
from mry_alert.transcription.adsb_prompt import build_adsb_prompt


def aircraft(
    *,
    hex_value: str = "abc123",
    registration: str | None = "N123AB",
    flight: str | None = None,
    operator: str | None = None,
    icao: str | None = None,
    iata: str | None = None,
) -> NearbyAircraft:
    return NearbyAircraft(
        hex=hex_value,
        registration=registration,
        flight=flight,
        operator_name=operator,
        icao_designator=icao,
        iata_code=iata,
        latitude=36.587,
        longitude=-121.843,
        altitude="ground",
        on_ground=True,
        ground_speed=5,
        seconds_since_seen=1,
        distance_nm=1,
    )


@pytest.mark.parametrize(
    ("icao", "flight", "operator"),
    [
        ("UAL", "UAL123", "United Airlines"),
        ("ASA", "ASA456", "Alaska Airlines"),
        ("AAL", "AAL789", "American Airlines"),
        ("DAL", "DAL321", "Delta Air Lines"),
        ("BAW", "BAW12", "British Airways"),
    ],
)
def test_known_scheduled_airlines_are_ignored(icao: str, flight: str, operator: str) -> None:
    result = TrafficFilter(TrafficFilterConfig()).evaluate_aircraft(
        aircraft(icao=icao, flight=flight, operator=operator)
    )
    assert result.decision == TrafficDecision.IGNORED


def test_generic_scheduled_airline_is_ignored_by_clear_operator_metadata() -> None:
    result = TrafficFilter(TrafficFilterConfig()).evaluate_aircraft(
        aircraft(icao="SIA", flight="SIA31", operator="Singapore Airlines")
    )
    assert result.decision == TrafficDecision.IGNORED
    assert "scheduled-airline" in result.reason


@pytest.mark.parametrize(
    "candidate",
    [
        aircraft(icao="JSX", flight="JSX123", operator="JSX"),
        aircraft(flight="JSX123"),
        aircraft(operator="JSX Air", registration="N525JS"),
        aircraft(flight="JSX456", registration="N525JS", operator=None, icao=None),
    ],
)
def test_jsx_is_always_allowed(candidate: NearbyAircraft) -> None:
    result = TrafficFilter(TrafficFilterConfig()).evaluate_aircraft(candidate)
    assert result.decision == TrafficDecision.ALLOWED_OVERRIDE
    assert result.reason == "JSX is explicitly allowed"


def test_explicit_jsx_allow_override_beats_deny_rules() -> None:
    config = TrafficFilterConfig(
        ignored_operator_names=["jsx"],
        ignored_icao_designators=["JSX"],
        ignored_callsign_prefixes=["JSX"],
    )
    result = TrafficFilter(config).evaluate_aircraft(
        aircraft(icao="JSX", flight="JSX123", operator="JSX")
    )
    assert result.decision == TrafficDecision.ALLOWED_OVERRIDE


def test_private_and_unknown_aircraft_are_not_filtered() -> None:
    traffic_filter = TrafficFilter(TrafficFilterConfig())
    private = traffic_filter.evaluate_aircraft(
        aircraft(flight="N123AB", operator="Monterey Aviation LLC")
    )
    unknown = traffic_filter.evaluate_aircraft(
        aircraft(hex_value="unknown", registration=None, flight=None, operator=None)
    )
    assert private.allowed
    assert private.decision == TrafficDecision.ALLOWED
    assert unknown.allowed
    assert unknown.decision == TrafficDecision.UNKNOWN
    assert traffic_filter.unknown_operator_count == 1


@pytest.mark.asyncio
async def test_filtered_airline_is_excluded_before_ranking_and_dynamic_prompt() -> None:
    blocked = aircraft(
        hex_value="united",
        registration="N111UA",
        flight="UAL123",
        operator="United Airlines",
        icao="UAL",
    )
    private = aircraft(hex_value="private", registration="N123AB")
    wrapper = TrafficFilteringProvider(
        MockNearbyAircraftProvider([blocked, private]),
        TrafficFilter(TrafficFilterConfig()),
    )

    eligible = await wrapper.nearby()

    assert eligible == [private]
    prompt = build_adsb_prompt(eligible)
    assert "N123AB" in prompt
    assert "N111UA" not in prompt


@pytest.mark.asyncio
async def test_filtered_airline_transcript_cannot_notify() -> None:
    published: list[object] = []

    async def publish(event: object) -> None:
        published.append(event)

    config = AppConfig()
    config.detection.destination_confirmation_delay_seconds = 0
    engine = DetectionEngine(
        config,
        MockNearbyAircraftProvider(
            [
                aircraft(
                    flight="UAL123",
                    operator="United Airlines",
                    icao="UAL",
                )
            ]
        ),
        publisher=publish,
    )
    event = TranscriptEvent(
        event_id="ual",
        timestamp=datetime.now(UTC),
        text="United one two three taxi to the Jet Center",
        source=config.liveatc.source_label,
    )

    result = await engine.process(event)
    await engine.flush_pending()

    assert result is None
    assert published == []
    assert event.traffic_filter_decision == TrafficDecision.IGNORED
    assert "airline callsign prefix" in " ".join(event.detection_reasons)


@pytest.mark.asyncio
async def test_jsx_flight_with_jet_center_intent_remains_eligible() -> None:
    config = AppConfig()
    candidate = aircraft(
        hex_value="jsx",
        registration="N525JS",
        flight="JSX123",
        operator="JSX",
        icao="JSX",
    )
    engine = DetectionEngine(config, MockNearbyAircraftProvider([candidate]))
    event = TranscriptEvent(
        event_id="jsx",
        timestamp=datetime.now(UTC),
        text="JSX one two three taxi via Alpha Echo to the Jet Center",
        source=config.liveatc.source_label,
    )

    result = await engine.process(event)

    assert isinstance(result, PendingDestinationEvent)
    assert result.registration == "N525JS"
    assert event.traffic_filter_decision == TrafficDecision.ALLOWED_OVERRIDE
    await engine.close()


def test_traffic_filter_metrics_count_unique_aircraft() -> None:
    traffic_filter = TrafficFilter(TrafficFilterConfig())
    united = aircraft(icao="UAL", flight="UAL123", operator="United Airlines")
    jsx = aircraft(hex_value="jsx", icao="JSX", flight="JSX123", operator="JSX")
    unknown = aircraft(hex_value="unknown", registration=None, flight=None, operator=None)
    for _ in range(2):
        traffic_filter.filter([united, jsx, unknown])
    assert traffic_filter.snapshot() == {
        "traffic_filter_enabled": True,
        "filtered_airline_count": 1,
        "allowed_override_count": 1,
        "unknown_operator_count": 1,
        "recent_traffic_filter_decision": TrafficDecision.UNKNOWN,
        "recent_traffic_filter_reason": "operator identity incomplete; default allow",
    }


def test_filter_log_is_explainable_and_not_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    TrafficFilter(TrafficFilterConfig()).evaluate_aircraft(
        aircraft(icao="UAL", flight="UAL123", operator="United Airlines")
    )

    record = next(item for item in caplog.records if "TRAFFIC FILTER" in item.message)
    assert record.levelno == logging.INFO
    assert "Aircraft: UAL123" in record.message
    assert "Operator: United Airlines" in record.message
    assert "Decision: ignored scheduled airline" in record.message
    assert "Reason: matched scheduled-airline ICAO designator UAL" in record.message
