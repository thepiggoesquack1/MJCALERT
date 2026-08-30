from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator


class MicrophoneAudioSource:
    def __init__(
        self, device: int | str | None, sample_rate: int = 16000, frame_duration_ms: int = 30
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Audio-device support is not installed. Run: pip install -e .[audio]"
            ) from exc
        self._sd = sd
        self.device = device
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_duration_ms // 1000

    async def frames(self) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

        def callback(indata: bytes, frames: int, time: object, status: object) -> None:
            del frames, time, status
            with contextlib.suppress(asyncio.QueueFull):
                loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

        with self._sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while True:
                yield await queue.get()
