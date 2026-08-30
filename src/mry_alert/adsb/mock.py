from mry_alert.models import NearbyAircraft


class MockNearbyAircraftProvider:
    def __init__(
        self, aircraft: list[NearbyAircraft] | None = None, error: Exception | None = None
    ) -> None:
        self.aircraft = aircraft or []
        self.error = error

    async def nearby(self) -> list[NearbyAircraft]:
        if self.error:
            raise self.error
        return self.aircraft
