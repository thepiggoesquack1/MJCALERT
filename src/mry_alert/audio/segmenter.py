from __future__ import annotations

import math
from collections import deque
from typing import Protocol

from mry_alert.config import AudioConfig


class VoiceDetector(Protocol):
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


class VadSegmenter:
    def __init__(self, config: AudioConfig, vad: VoiceDetector | None = None) -> None:
        if vad is None:
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError(
                    "VAD support is not installed. Run: pip install -e .[audio]"
                ) from exc
            vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.config = config
        self.vad = vad
        self.pre_roll_frames = max(1, math.ceil(config.pre_roll_ms / config.frame_duration_ms))
        self.silence_frames = max(1, math.ceil(config.end_silence_ms / config.frame_duration_ms))
        self.maximum_frames = max(
            1, int(config.max_transmission_seconds * 1000 / config.frame_duration_ms)
        )
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._active: list[bytes] = []
        self._silent = 0

    def add_frame(self, frame: bytes) -> bytes | None:
        speech = self.vad.is_speech(frame, self.config.sample_rate)
        if not self._active:
            if speech:
                self._active = list(self._pre_roll)
                self._active.append(frame)
                self._pre_roll.clear()
            else:
                self._pre_roll.append(frame)
            return None
        self._active.append(frame)
        self._silent = 0 if speech else self._silent + 1
        if self._silent >= self.silence_frames or len(self._active) >= self.maximum_frames:
            transmission = b"".join(self._active)
            self._active = []
            self._silent = 0
            return transmission
        return None

    def flush(self) -> bytes | None:
        if not self._active:
            return None
        transmission = b"".join(self._active)
        self._active = []
        self._silent = 0
        return transmission
