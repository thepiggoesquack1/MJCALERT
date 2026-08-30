from mry_alert.transcription.base import TranscriptionResult


class MockTranscriber:
    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = iter(transcripts)

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        del pcm, sample_rate
        return TranscriptionResult(
            text=next(self._transcripts),
            language_probability=1.0,
            average_log_probability=-0.2,
            no_speech_probability=0.01,
            segment_count=1,
            quality="high",
        )
