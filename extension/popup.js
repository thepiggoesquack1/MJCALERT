const $ = (id) => document.getElementById(id);

const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForNotificationResult(eventId) {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const {lastNotificationResult} = await chrome.storage.local.get({lastNotificationResult: null});
    if (lastNotificationResult?.eventId === eventId) return lastNotificationResult;
    await wait(250);
  }
  return null;
}

function renderAlerts(alerts) {
  $("alerts").replaceChildren(...alerts.map((event) => {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    const identity = event.registration || event.spoken_callsign || "Unresolved aircraft";
    strong.textContent = event.event_type === "destination_correction" ?
      `${identity} - corrected to ${event.corrected_destination || event.destination}` :
      event.event_type === "destination_cancelled" ? `${identity} - destination cancelled` :
        identity;
    const details = document.createElement("div");
    details.className = "muted";
    const sourceLabels = {
      spoken_full_registration: "Full spoken registration",
      spoken_callsign_adsb_match: "Spoken callsign matched with ADS-B",
      unique_suffix_adsb_match: "Abbreviated callsign recovered with ADS-B",
      fuzzy_adsb_recovery: "Noisy callsign conservatively recovered with ADS-B",
      unique_ground_candidate: "Only plausible taxiing aircraft (likely)",
      unresolved: "Aircraft identity unresolved"
    };
    const identification = sourceLabels[event.identification_source] ||
      "Aircraft identification source unavailable";
    details.textContent = `${event.confirmation_status || "confirmed"} / ${event.status} / ${
      Math.round(event.confidence * 100)}% / ${identification} / ${event.transcript_excerpt}`;
    li.append(strong, details);
    return li;
  }));
}

async function load() {
  const data = await chrome.storage.local.get({backendUrl: "http://127.0.0.1:8765", pairingToken: "", mute: false, minimumConfidence: 0.8, connectionStatus: "Disconnected", recentAlerts: [], notificationError: "", liveAtcStatus: "stopped", liveAtcDetail: "", liveAtcLastTranscript: ""});
  $("backendUrl").value = data.backendUrl;
  $("pairingToken").value = data.pairingToken;
  $("mute").checked = data.mute;
  $("minimumConfidence").value = data.minimumConfidence;
  $("confidenceValue").textContent = `${Math.round(data.minimumConfidence * 100)}%`;
  $("status").textContent = data.connectionStatus;
  $("liveAtcStatus").textContent = data.liveAtcStatus;
  $("liveAtcDetail").textContent = data.liveAtcDetail;
  $("lastTranscript").textContent = data.liveAtcLastTranscript ?
    `Last transcript: “${data.liveAtcLastTranscript}”` : "";
  if (data.notificationError) {
    $("feedback").textContent = `Notification failed: ${data.notificationError}`;
  } else if ($("feedback").textContent.startsWith("Notification failed:")) {
    $("feedback").textContent = "";
  }
  renderAlerts(data.recentAlerts);
}

$("minimumConfidence").addEventListener("input", () => $("confidenceValue").textContent = `${Math.round(Number($("minimumConfidence").value) * 100)}%`);
$("save").addEventListener("click", async () => {
  await chrome.storage.local.set({backendUrl: $("backendUrl").value.replace(/\/$/, ""), pairingToken: $("pairingToken").value.trim(), mute: $("mute").checked, minimumConfidence: Number($("minimumConfidence").value)});
  chrome.runtime.sendMessage({type: "settings-changed"});
  $("feedback").textContent = "Saved.";
});
$("test").addEventListener("click", async () => {
  try {
    $("feedback").textContent = "Sending test event…";
    await chrome.storage.local.set({notificationError: "", lastNotificationResult: null});
    const response = await fetch($("backendUrl").value.replace(/\/$/, "") + "/api/test-alert", {method: "POST", headers: {"X-Pairing-Token": $("pairingToken").value.trim()}});
    if (!response.ok) throw new Error(`Backend returned ${response.status}`);
    const event = await response.json();
    $("feedback").textContent = "Test event received by backend; waiting for Chrome…";
    const result = await waitForNotificationResult(event.event_id);
    if (!result) throw new Error("No notification result returned by the service worker within 5 seconds.");
    if (!result.ok) throw new Error(result.message || "Chrome rejected the notification.");
    $("feedback").textContent = result.active ?
      `Chrome created notification ${result.notificationId}; permission is ${result.permissionLevel}.` :
      `Chrome created notification ${result.notificationId}, but it is not active in Chrome's notification list.`;
  } catch (error) { $("feedback").textContent = `Test failed: ${error.message}`; }
});
$("startLiveAtc").addEventListener("click", async () => {
  $("liveAtcDetail").textContent = "Starting authorized capture…";
  const response = await chrome.runtime.sendMessage({type: "start-liveatc-capture"});
  $("liveAtcDetail").textContent = response?.ok ? response.message :
    `Start failed: ${response?.message || "Unknown error"}`;
});
$("stopLiveAtc").addEventListener("click", async () => {
  const response = await chrome.runtime.sendMessage({type: "stop-liveatc-capture"});
  $("liveAtcDetail").textContent = response?.ok ? "Capture stopped." :
    `Stop failed: ${response?.message || "Unknown error"}`;
});
chrome.storage.onChanged.addListener(load);
load();
