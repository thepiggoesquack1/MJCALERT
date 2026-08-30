# MRY Jet Center Alert MVP Plan

- [x] Create project configuration, domain models, and repository conventions
- [x] Implement normalization, callsign parsing, conversational context, and destination detection
- [x] Implement ADS-B providers and deterministic aircraft matching
- [x] Implement injectable detection engine and offline simulation fixtures
- [x] Implement authenticated FastAPI API, event storage, and WebSocket delivery
- [x] Implement file/microphone audio sources, VAD segmentation, and optional local Whisper adapter
- [x] Implement Manifest V3 Chrome extension with notifications and reconnect behavior
- [x] Add meaningful automated tests for detection, matching, failures, and authentication
- [x] Write user, architecture, testing, and future authorized-source documentation
- [x] Run and fix pytest, Ruff, mypy, and simulation acceptance checks

## Authorized LiveATC KMRY integration

- [x] Document the written permission and exact internal-use limitations
- [x] Add authenticated localhost PCM ingest and local transcription processing
- [x] Add user-initiated capture restricted to the authorized KMRY player page
- [x] Preserve player audio while forwarding transient PCM without recording by default
- [x] Add tests for URL restriction, authentication, configuration, and package integrity
- [x] Run all Python, typing, lint, JavaScript, simulation, and server checks

## Live KMRY recognition and observability hardening

- [x] Add readable per-transmission and real WebSocket delivery-result logging
- [x] Add KMRY-scoped ATC intent recovery and decoder repetition cleanup
- [x] Classify explicit, prompt-response, route, weak, and absent destination intent
- [x] Tune configurable LiveATC VAD defaults for shorter radio turns
- [x] Expose local Whisper quality metadata without claiming calibrated confidence
- [x] Add conservative ADS-B callsign, fuzzy, and unique-ground recovery
- [x] Preserve pending correction, contact isolation, authorization, and no-audio-storage rules
- [x] Add offline regressions for observed live failures and delivery observability
- [x] Upgrade configurable decoding to small.en with focused KMRY and cached ADS-B prompts
- [x] Add conservative radio-band preprocessing and 250/350/8 VAD tuning
- [x] Add model-comparison replay output without fabricated accuracy scores
- [x] Accept validated standard and Hugging Face faster-whisper model identifiers
- [x] Add local CPU/CUDA runtime diagnostics and explicit logged model fallback
- [x] Add same-segment multi-model comparison with WER/CER and aviation entity scoring
- [x] Add local dataset evaluation with downstream alert precision/recall and corrections
- [x] Add configurable per-model overrides, prompt toggles, RTF classes, and live backlog warnings
- [x] Add PySide6 Windows launcher, dashboard, safe console, tray, and redacted live logs
- [x] Add read-only backend status for extension, audio, speech, ADS-B, and notifications
- [x] Add PyInstaller onedir packaging, Windows build script, and optional shortcuts
- [x] Isolate model benchmarks in timed subprocesses with clean interruption and cleanup
- [x] Add persistent in-memory ADS-B histories, configurable geofences, movement states, and scoring
- [x] Make ADS-B correlation primary for identity while keeping speech mandatory for destination
- [x] Add conservative score/margin/conflict/staleness policies and offline ADS-B replay fixtures

## Local audio phrase and intent classification

- [x] Add optional multi-head classifier protocol, disabled/mock/rule/local implementations
- [x] Make classifier evidence primary only when explicitly enabled
- [x] Require destination, intent, and ADS-B resolution for classifier-driven notifications
- [x] Preserve pending evidence on noise, weak follow-ups, failures, and timeouts
- [x] Add default-off authorized local dataset collection and versioned metadata
- [x] Add human review and hard-negative queues with no automatic prediction-as-truth
- [x] Add deterministic augmentation, local baseline training, model export/load, and evaluation
- [x] Add desktop classifier status, review queue, and non-blocking train/evaluate actions
- [x] Add offline classifier, fusion, storage, metrics, timeout, and GUI tests
