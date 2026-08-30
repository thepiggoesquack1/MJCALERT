from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path


class WaveFileAudioSource:
    def __init__(self, path: Path, frame_duration_ms: int = 30) -> None:
        self.path = path
        self.sample_rate = 16000
        self.frame_duration_ms = frame_duration_ms

    async def frames(self) -> AsyncIterator[bytes]:
        with wave.open(str(self.path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
                raise ValueError("Audio WAV must be 16 kHz, mono, signed 16-bit PCM")
            samples = self.sample_rate * self.frame_duration_ms // 1000
            while frame := wav.readframes(samples):
                if len(frame) != samples * 2:
                    break
                yield frame
                await asyncio.sleep(0)
