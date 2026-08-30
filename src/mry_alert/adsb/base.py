from typing import Protocol

from mry_alert.models import NearbyAircraft


class NearbyAircraftProvider(Protocol):
    async def nearby(self) -> list[NearbyAircraft]: ...
