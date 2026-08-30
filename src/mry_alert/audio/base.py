from collections.abc import AsyncIterator
from typing import Protocol


class AudioSource(Protocol):
    sample_rate: int

    def frames(self) -> AsyncIterator[bytes]: ...
