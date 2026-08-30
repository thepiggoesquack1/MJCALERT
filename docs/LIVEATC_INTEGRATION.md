# Authorized LiveATC KMRY integration

## Permission record

LiveATC granted written permission on July 18, 2026 at 9:59 AM Pacific via `liveatc@support.liveatc.net`. The permission responded to a request that described this exact internal Monterey Jet Center tool: locally recognizing KMRY Ground communications, correlating callsigns with public ADS-B information, and showing operational-awareness notifications to line service staff.

The authorized player is:

`https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry`

The original request committed that the tool would be internal, noncommercial, would not redistribute or publicly rebroadcast LiveATC audio, and would not record it. LiveATC approved that purpose while requiring compliance with its remaining terms and expressly excepting the otherwise conflicting software-use policy. Keep the original email and complete headers outside this repository as the authoritative permission evidence.

## Enforced scope

The extension requires an explicit user click and captures only an active HTTPS LiveATC player tab whose host is `www.liveatc.net`, path is `/hlisten.php`, and `mount` and `icao` query parameters both equal `kmry`. It does not discover, extract, embed, or log an underlying stream URL. It does not scrape LiveATC pages, access archives, or capture any other feed.

Captured audio is converted in memory to 16 kHz mono signed 16-bit PCM and sent only to the authenticated backend on loopback. The backend uses it transiently for VAD and local speech recognition. LiveATC audio is never saved, even when the general `audio.save_debug_audio` setting is enabled. The existing debug-audio option applies only to user-selected file and microphone workflows.

The captured player audio is routed back to the default output so the user can continue hearing it. Capture stops on explicit user request, when the tab/stream ends, or when the offscreen document or extension is unloaded.

## Architecture boundary

Chrome 116+ service worker, after a popup user gesture
→ validate exact authorized player URL
→ request `chrome.tabCapture` stream ID
→ offscreen document consumes the audio-only stream
→ AudioWorklet preserves playback and converts samples
→ authenticated `ws://127.0.0.1:8765/ws/audio`
→ 30 ms PCM framing and WebRTC VAD
→ local faster-whisper transcription
→ existing detection, ADS-B matching, event store, and notification path

No LiveATC-specific URL or network request exists in the Python backend. Revocation or material changes to permission must disable `liveatc.enabled` and remove/stop the capture workflow until the scope is reviewed again.
