from __future__ import annotations

import math
from typing import cast

import numpy as np

from mry_alert.config import AudioPreprocessingConfig


def _high_pass(samples: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
    if cutoff <= 0 or samples.size < 2:
        return samples
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = rc / (rc + 1.0 / sample_rate)
    output = np.empty_like(samples)
    output[0] = samples[0]
    for index in range(1, samples.size):
        output[index] = alpha * (output[index - 1] + samples[index] - samples[index - 1])
    return output


def _low_pass(samples: np.ndarray, cutoff: float, sample_rate: int) -> np.ndarray:
    if cutoff <= 0 or cutoff >= sample_rate / 2 or samples.size < 2:
        return samples
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = (1.0 / sample_rate) / (rc + 1.0 / sample_rate)
    output = np.empty_like(samples)
    output[0] = samples[0]
    for index in range(1, samples.size):
        output[index] = output[index - 1] + alpha * (samples[index] - output[index - 1])
    return output


def preprocess_pcm16(pcm: bytes, sample_rate: int, config: AudioPreprocessingConfig) -> bytes:
    if not config.enabled:
        return pcm
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"Audio preprocessing expects {config.sample_rate} Hz mono PCM; got {sample_rate} Hz"
        )
    if len(pcm) % 2:
        raise ValueError("16-bit PCM byte length must be even")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return pcm
    samples -= float(np.mean(samples))
    samples = _high_pass(samples, config.high_pass_hz, sample_rate)
    samples = _low_pass(samples, config.low_pass_hz, sample_rate)
    peak = float(np.max(np.abs(samples)))
    if config.normalize and peak > 1e-8:
        target = 10.0 ** (config.target_peak_dbfs / 20.0)
        samples *= target / peak
    if config.limiter:
        samples = np.clip(samples, -0.98, 0.98)
    return cast(bytes, np.rint(samples * 32767.0).astype(np.int16).tobytes())
