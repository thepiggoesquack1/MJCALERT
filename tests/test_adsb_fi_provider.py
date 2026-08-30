from __future__ import annotations

import httpx
import pytest

from mry_alert.adsb.adsb_fi import AdsbFiNearbyAircraftProvider
from mry_alert.adsb.adsb_lol import AdsbLolNearbyAircraftProvider
from mry_alert.adsb.factory import create_nearby_aircraft_provider
from mry_alert.config import AdsbConfig, AirportConfig


@pytest.mark.asyncio
async def test_adsb_fi_uses_documented_v3_nearby_endpoint_and_maps_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/lat/36.587/lon/-121.843/dist/5"
        return httpx.Response(
            200,
            json={
                "ac": [
                    {
                        "hex": "abc123",
                        "r": "N123AB",
                        "flight": "TEST1",
                        "t": "C25B",
                        "lat": 36.59,
                        "lon": -121.84,
                        "alt_baro": "ground",
                        "gs": 8,
                        "seen": 1,
                        "category": "A2",
                    }
                ]
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AdsbFiNearbyAircraftProvider(
            AirportConfig(),
            AdsbConfig(
                provider="adsb_fi",
                base_url="https://opendata.adsb.fi/api",
                polling_interval_seconds=0,
            ),
            client,
        )
        aircraft = await provider.nearby()

    assert aircraft[0].registration == "N123AB"
    assert aircraft[0].aircraft_type_code == "C25B"
    assert aircraft[0].aircraft_category == "A2"
    assert aircraft[0].on_ground is True
    assert provider.last_error is None
    assert provider.last_success_at is not None


@pytest.mark.asyncio
async def test_adsb_fi_does_not_probe_undocumented_fallback_routes() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(502, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = AdsbFiNearbyAircraftProvider(
            AirportConfig(),
            AdsbConfig(
                provider="adsb_fi",
                base_url="https://opendata.adsb.fi/api",
                polling_interval_seconds=0,
            ),
            client,
        )
        assert await provider.nearby() == []

    assert requested_paths == ["/api/v3/lat/36.587/lon/-121.843/dist/5"]
    assert provider.last_error is not None


def test_provider_factory_selects_configured_adapter() -> None:
    airport = AirportConfig()
    adsb_fi = create_nearby_aircraft_provider(
        airport,
        AdsbConfig(provider="adsb_fi", base_url="https://opendata.adsb.fi/api"),
    )
    adsb_lol = create_nearby_aircraft_provider(airport, AdsbConfig())

    assert isinstance(adsb_fi, AdsbFiNearbyAircraftProvider)
    assert isinstance(adsb_lol, AdsbLolNearbyAircraftProvider)
