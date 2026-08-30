from __future__ import annotations

from mry_alert.adsb.adsb_lol import AdsbLolNearbyAircraftProvider


class AdsbFiNearbyAircraftProvider(AdsbLolNearbyAircraftProvider):
    """Nearby-aircraft adapter for the authorized ADSB.fi open-data API."""

    def _endpoint_urls(self) -> tuple[str, str | None]:
        url = (
            f"{self.config.base_url.rstrip('/')}/v3/lat/{self.airport.latitude}"
            f"/lon/{self.airport.longitude}/dist/{self.airport.adsb_radius_nm}"
        )
        return url, None

    def _request_headers(self) -> dict[str, str]:
        return {
            "User-Agent": (
                "MRY-Jet-Center-Alert/0.1 (authorized local advisory use; "
                "ADS-B data courtesy of https://adsb.fi/)"
            )
        }
