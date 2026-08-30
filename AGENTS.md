# Project conventions

- Keep runtime cloud-free: speech recognition is local and optional.
- Never add LiveATC access/capture code without documented written permission.
- Keep simulation and unit tests free of model downloads, audio hardware, and network access.
- Put configurable behavior in `AppConfig`; use dependency injection for external providers.
- Use UTC-aware timestamps, `pathlib`, typed models, and explainable scoring reasons.
- Never log pairing tokens or save raw audio unless explicitly enabled in configuration.
- Before handoff run: `pytest`, `ruff check .`, and `mypy src`.
