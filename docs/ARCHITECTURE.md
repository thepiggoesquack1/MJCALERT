# Architecture

## Event flow

```text
Audio source (WAV, microphone, or authorized KMRY player tab)
  -> voice activity detection (16 kHz PCM, pre-roll, silence cutoff)
  -> configurable DC removal, 275 Hz high-pass, 3.8 kHz low-pass, normalization, limiter
  -> optional local narrow audio destination/intent classifier
  -> local faster-whisper transcription (operator text and secondary evidence)
  -> deterministic transcript normalization
  -> authorized-source ATC intent recovery and repetitive-tail cleanup
  -> phraseology/turn-taking speaker-role inference (controller, pilot, or unknown)
  -> aviation callsign parser and short-lived contact context
  -> conservative destination evidence and competing-destination parsing
  -> cached nearby ADS-B lookup and explainable aircraft matching
  -> per-contact PendingDestinationEvent (non-blocking confirmation timer)
  -> confirmed AlertEvent, or corrected/cancelled state
  -> authenticated localhost FastAPI WebSocket
  -> Manifest V3 service-worker WebSocket (Chrome 116+ keepalive)
  -> Chrome desktop notification and recent-alert list
```

Simulation injects transcript fixtures and a `MockNearbyAircraftProvider` after the audio/transcription stages. It therefore exercises the same detection engine without a model, hardware, or network.

## Boundaries

- `audio`: generic `AudioSource`, WAV/device adapters, and WebRTC VAD transmission segmentation.
- `transcription`: injectable `Transcriber`; mock and optional local faster-whisper implementations.
- `audio_classifier`: injectable multi-head classifier, reviewed dataset, augmentation, local
  baseline training, evaluation, and portable model loading.
- `detection`: pure normalization/parsing, contact context, destination evidence, matching, and orchestration.
- `adsb`: injectable provider protocol, deterministic mock, and cached/backing-off ADSB.fi and
  adsb.lol adapters selected through configuration.
- `server`: token management, bounded in-memory/JSONL event history, REST status, and WebSocket clients.
- `extension`: localhost connection, preferences, deduplication, safe DOM rendering, and notifications.

The authorized browser path adds a user-gesture boundary: the service worker validates the exact KMRY player URL, requests a `tabCapture` stream ID, and passes it to an offscreen document. An AudioWorklet preserves playback, converts/resamples to 16 kHz mono PCM, and forwards only transient chunks to `/ws/audio`. The backend never contacts LiveATC and never persists those chunks.

Important values live in `AppConfig` and `config.example.yaml`. Domain messages use typed Pydantic models and UTC-aware timestamps. External providers are injected, allowing every unit test to stay offline.

## Classifier and evidence fusion

`AudioIntentClassifier` accepts transient PCM plus a typed context and returns separate destination,
intent, correction, and noise confidences. Implementations include disabled, mock, conservative
feature-only noise rejection, and a portable trainable local baseline. The baseline extracts
deterministic radio-audio spectral/time features and trains separate class-balanced destination and
intent centroids. It is deliberately small, CPU-capable, framework-independent, and suitable as a
Phase 1/2 baseline—not a claim of production semantic accuracy.

With classifier-primary mode enabled, a Monterey alert requires destination and route/parking
scores above threshold and a resolved ADS-B identity. Del Monte suppresses the Monterey path.
Corrections operate through the existing per-contact correction state. Noise, inference timeout,
model failure, and weak output create no new event and do not mutate an existing strong pending
event. ADS-B can resolve identity but can never invent destination. Whisper-only behavior is behind
an explicit fallback flag and remains the unchanged behavior while the classifier is disabled.

Dataset collection is a separate, default-off boundary. When explicitly enabled it writes mono WAV
and versioned JSON metadata to `pending/`. Human review moves clips to `reviewed/`, `rejected/`, or
`hard_negatives/`; only records marked `reviewed: true` and `label_source: human_review` can train.
Raw classifier scores are optional, pairing-token patterns are redacted, and no network or upload
path exists.

Speaker role is not an acoustic identity. The deterministic classifier considers request/command phraseology, callsign placement, readback structure, multiple addressed callsigns, and whether a short destination response follows an unambiguous controller parking prompt. It records a bounded confidence and explainable reasons. No voice samples, speaker embeddings, or biometric comparisons are created. Conflicting or weak evidence remains `unknown`.

Intent recovery, destination intent, decoder-artifact cleanup, and aircraft matching remain separate modules. Intent recovery is enabled only when the `TranscriptEvent.source` equals the configured authorized LiveATC source label. Route intent requires a high-confidence destination, KMRY ground source, ground-route cues, and a structural connection to the destination. Business mentions remain weak. Artifact cleanup changes only detector text and records what repeated tail was removed.

The external VAD targets one push-to-talk turn per request; faster-whisper internal VAD and previous-text conditioning are disabled by default. A non-blocking ADS-B refresh task builds a bounded dynamic prompt without delaying transcription. Audio sources already enforce/resample to 16 kHz mono PCM; preprocessing validates that contract and never replaces original debug audio.

Model selection accepts standard faster-whisper names and validated Hugging Face
`owner/repository` identifiers. Global speech settings are copied and optional validated per-model
overrides are applied before initialization. Initialization is lazy for the server, so unrelated
commands never download a model. Explicit fallback is handled inside the local transcriber and is
always logged; it is disabled unless both the enable flag and fallback model are configured.

Offline comparison segments a WAV once and reuses those exact PCM transmissions for every model.
It records raw decoder output plus the existing normalization, artifact filter, speaker-role,
callsign, destination, intent, ADS-B matching, correction, and alert-decision stages. Evaluation
never publishes to the Chrome WebSocket. Live ingestion separates receipt/segmentation from an
unbounded transcription worker queue, exposes backlog, and warns on configurable rolling RTF rather
than silently dropping completed segments.

Each comparison candidate executes in a dedicated subprocess with a hard timeout. PCM segments and
validated configuration are passed through a temporary local request file; results return through
a separate file. Worker termination releases all native CUDA/CTranslate2 allocations even when a
decoder call cannot be cancelled. Within a successful worker, the faster-whisper generator is
materialized exactly once and model close/unload hooks plus garbage collection run before exit.

`AdsbContactTracker` keeps bounded, in-memory histories keyed by ICAO hex. `AdsbCorrelator` derives
distance trend and recent-landing/movement state, applies configured point weights, and requires a
minimum winner score and margin. Detection intent still originates exclusively from speech. The
correlator may supply identity without a spoken callsign, but it cannot create a destination event;
stale, implausible, conflicting, and ambiguous identities are withheld. Geofences are configuration
objects and unverified Jet Center coordinates are never embedded in Python.

## Scoring and safety behavior

Exact destination phrases are preferred. Fuzzy destination similarity cannot trigger alone. A destination must be paired with taxi/parking language or uniquely linked to a recent parking question. Full registration matches carry the largest aircraft score; suffixes require uniqueness plus ground, taxi-speed, and recency evidence. Proximity alone never establishes identity. Similar top candidates are ambiguous, and ambiguous/unresolved events are not emitted by default.

The legacy `alert_on_any_destination_mention` setting is retained for configuration compatibility,
but does not turn a bare destination mention into an arrival. Explicit pilot-arrival phraseology or
an unambiguous parking-prompt response is required. Controller-only destination instructions and
pilot current-location/departure reports become ignored or unresolved rather than arrivals.

When `immediate_notification_on_clear_ground_match` is also enabled, a confirmed ADS-B winner with
a registration is promoted from pending to confirmed in the same processing turn. The confirmed
record remains inside the normal correction window, so a linked destination change can still issue
a correction.

Each contact moves through `destination_unknown`, `destination_pending`, `destination_confirmed`, and either `destination_corrected` or `destination_cancelled`. The radio-correction delay and the longer ADS-B correlation window are separate: after the correction delay, unresolved arrivals can gather later ADS-B observations until the correlation deadline. A same-contact correction cancels that task under a state lock. The parser chooses the latest explicit configured destination; conservative correction markers and callsign/awaiting-parking context prevent another aircraft or unrelated “no” from cancelling the event. Confirmed events remain correctable for 20 seconds by default. A late correction is a new `destination_correction` event referencing the original event ID; an early correction updates the pending event without ever publishing an arrival.

A callsign-free destination reply inherits a contact only when that contact was asked for parking/destination within the context window and no newer competing contact or second awaiting prompt makes the turn ambiguous. Strong controller-role structure prevents a controller instruction mentioning Monterey Jet Center from becoming a pilot arrival request. Role inference is supporting context only: callsign/contact association and destination evidence remain independently required, and corrections still require a contact link plus correction or replacement-destination evidence.

Duplicate suppression keys qualifying confirmed events by registration (or spoken form when unresolved) and destination for the configured window. Only one pending destination exists per contact. Provider exceptions produce an unresolved match rather than stopping monitoring.

Aircraft matching filters ADS-B candidates by configured distance, recency, ground/low-altitude state, and taxi speed before identity scoring. Exact speech-to-registration evidence has priority, followed by flight callsign, unique suffix, conservative fuzzy spoken forms, and finally a unique ground candidate. Fuzzy and unique-ground recovery are never confirmed. A strong authorized route may create a pending event with ambiguous or unresolved identity; this does not relax destination or correction safeguards.

`WebSocketManager.broadcast` returns connected, delivered, and failed counts. The application logs delivery only from that result, with an application-scoped event-ID tracker preventing duplicate “sent” messages. The detector records one decision on each `TranscriptEvent`; audio ingestion emits one operational transmission block after processing. Secrets and raw PCM are never passed to the operational formatter.

## Local security and storage

The server defaults to loopback only. A cryptographically random bearer-style pairing token is generated on first run. REST test events use `X-Pairing-Token`; WebSockets use a token query parameter because browser WebSocket APIs cannot set arbitrary headers. Tokens are compared with a timing-safe function and never logged. Pairing tokens, event logs, recordings, caches, and model data are ignored by Git.
