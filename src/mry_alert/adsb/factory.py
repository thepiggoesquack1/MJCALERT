from __future__ import annotations

import httpx

from mry_alert.adsb.adsb_fi import AdsbFiNearbyAircraftProvider
from mry_alert.adsb.adsb_lol import AdsbLolNearbyAircraftProvider
from mry_alert.adsb.base import NearbyAircraftProvider
from mry_alert.config import AdsbConfig, AirportConfig


def create_nearby_aircraft_provider(
    airport: AirportConfig,
    config: AdsbConfig,
    client: httpx.AsyncClient | None = None,
) -> NearbyAircraftProvider:
    if config.provider == "adsb_fi":
        return AdsbFiNearbyAircraftProvider(airport, config, client)
    if config.provider == "adsb_lol":
        return AdsbLolNearbyAircraftProvider(airport, config, client)
    raise ValueError(f"Unsupported ADS-B provider: {config.provider}")
