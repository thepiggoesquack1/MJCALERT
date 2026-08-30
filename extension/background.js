let socket;
let reconnectTimer;
let keepAliveTimer;
let reconnectDelay = 1000;
let creatingOffscreenDocument;
let connectPromise;

const AUTHORIZED_LIVEATC_URL = "https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry";

function isAuthorizedLiveAtcUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" &&
      url.hostname === "www.liveatc.net" &&
      url.pathname === "/hlisten.php" &&
      url.searchParams.get("mount")?.toLowerCase() === "kmry" &&
      url.searchParams.get("icao")?.toLowerCase() === "kmry";
  } catch {
    return false;
  }
}

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL("offscreen.html")]
  });
  if (contexts.length > 0) return;
  if (!creatingOffscreenDocument) {
    creatingOffscreenDocument = chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["USER_MEDIA"],
      justification: "Process the user-authorized KMRY LiveATC player audio locally."
    }).finally(() => {
      creatingOffscreenDocument = null;
    });
  }
  await creatingOffscreenDocument;
}

async function startLiveAtcCapture() {
  const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
  if (!tab?.id || !isAuthorizedLiveAtcUrl(tab.url || "")) {
    throw new Error(`Open the authorized KMRY player first: ${AUTHORIZED_LIVEATC_URL}`);
  }
  const settings = await chrome.storage.local.get({
    backendUrl: "http://127.0.0.1:8765",
    pairingToken: ""
  });
  if (!settings.pairingToken) throw new Error("Save the backend pairing token first.");
  await ensureOffscreenDocument();
  const streamId = await chrome.tabCapture.getMediaStreamId({targetTabId: tab.id});
  const response = await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "start-liveatc-capture",
    data: {
      streamId,
      tabId: tab.id,
      backendUrl: settings.backendUrl,
      pairingToken: settings.pairingToken
    }
  });
  if (!response?.ok) throw new Error(response?.message || "Offscreen audio capture did not start.");
  return response;
}

async function stopLiveAtcCapture() {
  await ensureOffscreenDocument();
  const response = await chrome.runtime.sendMessage({
    target: "offscreen",
    type: "stop-liveatc-capture"
  });
  return response || {ok: true};
}

function createNotification(notificationId, options) {
  return new Promise((resolve, reject) => {
    try {
      chrome.notifications.create(notificationId, options, (createdId) => {
        const lastError = chrome.runtime.lastError;
        if (lastError) {
          const error = new Error(lastError.message);
          console.error("MRY notification creation failed:", error);
          reject(error);
          return;
        }
        if (!createdId) {
          const error = new Error("Chrome did not return a notification ID.");
          console.error("MRY notification creation failed:", error);
          reject(error);
          return;
        }
        resolve(createdId);
      });
    } catch (error) {
      console.error("MRY notification creation threw an exception:", error);
      reject(error);
    }
  });
}

async function rememberAndNotify(event) {
  console.info("MRY destination event received:", event.event_id,
    event.event_type, event.confirmation_status);
  const settings = await chrome.storage.local.get({
    recentAlerts: [], mute: false, minimumConfidence: 0.8, seenEventIds: []
  });
  const isPending = event.confirmation_status === "pending";
  const isCorrection = event.event_type === "destination_correction" ||
    event.event_type === "destination_cancelled";
  let replaced = false;
  const recentAlerts = settings.recentAlerts.map((stored) => {
    if (stored.event_id === event.event_id ||
        (event.original_event_id && stored.event_id === event.original_event_id)) {
      replaced = true;
      return {...stored, ...event, event_id: stored.event_id,
        correction_event_id: event.original_event_id ? event.event_id : undefined};
    }
    return stored;
  });
  if (!replaced) recentAlerts.unshift(event);
  await chrome.storage.local.set({recentAlerts: recentAlerts.slice(0, 10)});
  if (isPending) {
    console.info("Pending destination stored without desktop notification:", event.event_id);
    return {status: "pending_not_notified"};
  }
  if (settings.seenEventIds.includes(event.event_id)) {
    return {status: "duplicate_ignored"};
  }
  const seenEventIds = [event.event_id, ...settings.seenEventIds].slice(0, 100);
  await chrome.storage.local.set({seenEventIds});
  if (!event.test && (settings.mute ||
      (!isCorrection && event.confidence < Number(settings.minimumConfidence)))) {
    console.info("MRY notification suppressed by user settings:", event.event_id);
    await chrome.storage.local.set({lastNotificationResult: {
      eventId: event.event_id, ok: false, suppressed: true,
      message: "Notification suppressed by mute or minimum-confidence setting."
    }});
    return {status: "suppressed"};
  }
  const permissionLevel = await chrome.notifications.getPermissionLevel();
  if (permissionLevel !== "granted") {
    throw new Error(`Chrome notification permission is ${permissionLevel}.`);
  }
  const ambiguous = event.status === "likely" || event.status === "ambiguous" ||
    event.status === "unresolved";
  const aircraft = event.registration || event.spoken_callsign || "Unknown aircraft";
  const typeName = event.aircraft_type_name || event.aircraft_type ||
    "Aircraft type unknown";
  const identity = `${aircraft} — ${typeName}`;
  const title = event.test ? `TEST — ${identity}` : identity;
  const alternatives = event.alternative_registrations?.length ?
    `\nPossible matches: ${event.alternative_registrations.join(", ")}` : "";
  const identificationLabels = {
    spoken_full_registration: "full spoken registration",
    spoken_callsign_adsb_match: "spoken callsign plus ADS-B",
    unique_suffix_adsb_match: "unique callsign suffix plus ADS-B",
    fuzzy_adsb_recovery: "conservative fuzzy ADS-B recovery",
    unique_ground_candidate: "only plausible taxiing aircraft (likely)",
    adsb_correlation: "ADS-B movement and proximity correlation",
    unresolved: "unresolved aircraft identity"
  };
  const identification = identificationLabels[event.identification_source] ||
    "aircraft identity source unavailable";
  const correctionMessage = event.event_type === "destination_correction" ?
    `${identity} is no longer expected at Monterey Jet Center. Updated destination: ${
      event.corrected_destination || event.destination}.` :
    `${identity} is no longer expected at Monterey Jet Center.`;
  const destination = event.destination || "Monterey Jet Center";
  const normalMessage = `${ambiguous ? "Possible arrival" : "Going to " + destination}\n` +
    `Confidence: ${Math.round(event.confidence * 100)}%${
    alternatives}\nMatched from: ${identification}\n\u201c${event.transcript_excerpt}\u201d`;
  const createdId = await createNotification(event.event_id, {
    type: "basic",
    iconUrl: chrome.runtime.getURL("icons/icon128.png"),
    title,
    message: isCorrection ? correctionMessage : normalMessage,
    priority: event.test || isCorrection ? 2 : 1,
    requireInteraction: Boolean(event.test || isCorrection),
    silent: false
  });
  const activeNotifications = await chrome.notifications.getAll();
  const result = {
    eventId: event.event_id,
    notificationId: createdId,
    ok: true,
    permissionLevel,
    active: Object.hasOwn(activeNotifications, createdId),
    createdAt: new Date().toISOString()
  };
  console.info("MRY notification created:", result);
  await chrome.storage.local.set({notificationError: "", lastNotificationResult: result});
  return {status: "delivered"};
}

function scheduleReconnect() {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 2, 30000);
}

async function connect(force = false) {
  if (!force && socket &&
      (socket.readyState === WebSocket.OPEN ||
       socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  if (connectPromise) {
    await connectPromise;
    return force ? connect(true) : undefined;
  }
  connectPromise = connectOnce().finally(() => {
    connectPromise = null;
  });
  return connectPromise;
}

async function connectOnce() {
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  clearInterval(keepAliveTimer);
  if (socket) {
    socket.onclose = null;
    socket.close();
    socket = null;
  }
  const settings = await chrome.storage.local.get({
    backendUrl: "http://127.0.0.1:8765", pairingToken: ""
  });
  if (!settings.pairingToken) {
    await chrome.storage.local.set({connectionStatus: "Pairing token required"});
    return;
  }
  const url = settings.backendUrl.replace(/^http/, "ws").replace(/\/$/, "") +
    "/ws?token=" + encodeURIComponent(settings.pairingToken);
  const currentSocket = new WebSocket(url);
  socket = currentSocket;
  currentSocket.onopen = async () => {
    if (socket !== currentSocket) return;
    reconnectDelay = 1000;
    await chrome.storage.local.set({connectionStatus: "Connected"});
    // Chrome 116+ resets the service-worker idle timer on WebSocket activity.
    keepAliveTimer = setInterval(() => {
      if (currentSocket.readyState === WebSocket.OPEN) currentSocket.send("keepalive");
    }, 20000);
  };
  currentSocket.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    if (event.type !== "heartbeat" && event.event_id) {
      try {
        const outcome = await rememberAndNotify(event);
        if (currentSocket.readyState === WebSocket.OPEN) {
          currentSocket.send(JSON.stringify({
            type: "notification_delivery_ack",
            event_id: event.event_id,
            status: outcome.status
          }));
        }
      } catch (error) {
        console.error("MRY alert processing failed:", error);
        await chrome.storage.local.set({
          notificationError: error.message,
          lastNotificationResult: {
            eventId: event.event_id,
            ok: false,
            message: error.message,
            createdAt: new Date().toISOString()
          }
        });
        if (currentSocket.readyState === WebSocket.OPEN) {
          currentSocket.send(JSON.stringify({
            type: "notification_delivery_ack",
            event_id: event.event_id,
            status: "failed"
          }));
        }
      }
    }
  };
  currentSocket.onerror = () => currentSocket.close();
  currentSocket.onclose = async (event) => {
    if (socket !== currentSocket) return;
    socket = null;
    clearInterval(keepAliveTimer);
    if (event.code === 4403) {
      await chrome.storage.local.set({
        connectionStatus: "Pairing token rejected — paste the token from the active backend"
      });
      console.error("MRY WebSocket authentication failed: pairing token rejected.");
      return;
    }
    await chrome.storage.local.set({connectionStatus: "Disconnected"});
    scheduleReconnect();
  };
}

chrome.runtime.onInstalled.addListener(connect);
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target === "offscreen") return false;
  if (message.type === "settings-changed") {
    connect(true);
    return false;
  }
  if (message.type === "liveatc-capture-status") {
    chrome.storage.local.set({
      liveAtcStatus: message.status,
      liveAtcDetail: message.detail || "",
      liveAtcLastTranscript: message.transcript || ""
    });
    return false;
  }
  if (message.type === "start-liveatc-capture") {
    startLiveAtcCapture()
      .then(sendResponse)
      .catch((error) => {
        console.error("MRY LiveATC capture start failed:", error);
        sendResponse({ok: false, message: error.message});
      });
    return true;
  }
  if (message.type === "stop-liveatc-capture") {
    stopLiveAtcCapture()
      .then(sendResponse)
      .catch((error) => sendResponse({ok: false, message: error.message}));
    return true;
  }
  return false;
});
chrome.tabCapture.onStatusChanged.addListener((info) => {
  if (info.status === "stopped" || info.status === "error") {
    chrome.storage.local.set({
      liveAtcStatus: info.status,
      liveAtcDetail: `Chrome tab capture ${info.status}.`
    });
  }
});
connect();
