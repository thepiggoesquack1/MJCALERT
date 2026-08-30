from datetime import UTC, datetime

import httpx
import pytest

from mry_alert.adsb.adsb_lol import AdsbLolNearbyAircraftProvider
from mry_alert.config import AdsbConfig, AirportConfig


@pytest.mark.asyncio
async def test_provider_uses_nearby_endpoint_and_maps_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/lat/36.587/lon/-121.843/dist/5"
        return httpx.Response(
            200,
            json={
                "ac": [
                    {
                        "hex": "abc123",
                        "r": " N123AB ",
                        "flight": " TEST1 ",
                        "ownOp": "Test Operator",
                        "icao": "TST",
                        "t": "C525",
                        "lat": 36.59,
                        "lon": -121.84,
                        "alt_baro": "ground",
                        "gs": 7,
                        "seen": 2,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://example.invalid"
    ) as client:
        provider = AdsbLolNearbyAircraftProvider(
            AirportConfig(), AdsbConfig(base_url="https://example.invalid"), client
        )
        aircraft = await provider.nearby()
        cached = await provider.nearby()

    assert aircraft == cached
    assert aircraft[0].registration == "N123AB"
    assert aircraft[0].flight == "TEST1"
    assert aircraft[0].operator_name == "Test Operator"
    assert aircraft[0].icao_designator == "TST"
    assert aircraft[0].aircraft_type == "C525"
    assert aircraft[0].aircraft_type_code == "C525"
    assert aircraft[0].aircraft_type_source == "adsb_provider"
    assert aircraft[0].aircraft_type_confidence == 0.95
    assert aircraft[0].on_ground
    assert aircraft[0].source_timestamp is not None


@pytest.mark.asyncio
async def test_provider_retries_documented_point_route_after_server_error() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if "/v2/lat/" in request.url.path:
            return httpx.Response(502, request=request)
        return httpx.Response(
            200,
            json={
                "ac": [
                    {
                        "hex": "abc123",
                        "r": "N123AB",
                        "alt_baro": "ground",
                        "seen": 2,
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AdsbLolNearbyAircraftProvider(
            AirportConfig(), AdsbConfig(polling_interval_seconds=0), client
        )
        aircraft = await provider.nearby()

    assert requested_paths == [
        "/v2/lat/36.587/lon/-121.843/dist/5",
        "/v2/point/36.587/-121.843/5",
    ]
    assert aircraft[0].registration == "N123AB"
    assert provider.last_error is None
    assert provider.last_success_at is not None
    assert aircraft[0].source_timestamp is not None
    assert aircraft[0].source_timestamp < datetime.now(UTC)


@pytest.mark.asyncio
async def test_provider_returns_cache_on_temporary_failure() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ConnectTimeout("offline")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AdsbLolNearbyAircraftProvider(
            AirportConfig(), AdsbConfig(polling_interval_seconds=0), client
        )
        assert await provider.nearby() == []
