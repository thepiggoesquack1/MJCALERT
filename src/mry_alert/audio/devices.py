def list_audio_devices() -> str:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Audio-device support is not installed. Run: pip install -e .[audio]"
        ) from exc
    return str(sd.query_devices())
