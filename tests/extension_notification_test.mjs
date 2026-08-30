import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {URL} from "node:url";

const source = fs.readFileSync(new URL("../extension/background.js", import.meta.url), "utf8");
let runtimeLastError = null;
let createImplementation = (_id, _options, callback) => callback("created-id");
let lastCreateOptions = null;
const loggedErrors = [];
const storageState = {
  pairingToken: "",
  recentAlerts: [],
  mute: true,
  minimumConfidence: 1,
  seenEventIds: [],
};

const chrome = {
  notifications: {
    create: (id, options, callback) => {
      lastCreateOptions = options;
      createImplementation(id, options, callback);
    },
    getAll: async () => ({"created-id": true}),
    getPermissionLevel: async () => "granted",
  },
  runtime: {
    get lastError() {
      return runtimeLastError;
    },
    getURL: (path) => `chrome-extension://test/${path}`,
    onInstalled: {addListener: () => {}},
    onStartup: {addListener: () => {}},
    onMessage: {addListener: () => {}},
    getContexts: async () => [],
  },
  storage: {
    local: {
      get: async (defaults) => ({...defaults, ...storageState}),
      set: async (values) => Object.assign(storageState, values),
    },
  },
  offscreen: {createDocument: async () => {}},
  tabs: {query: async () => []},
  tabCapture: {
    getMediaStreamId: async () => "stream-id",
    onStatusChanged: {addListener: () => {}},
  },
};

const context = vm.createContext({
  chrome,
  console: {error: (...args) => loggedErrors.push(args), info: () => {}},
  clearInterval: () => {},
  clearTimeout: () => {},
  encodeURIComponent,
  Error,
  Promise,
  URL,
  setInterval: () => 1,
  setTimeout: () => 1,
});
vm.runInContext(source, context);

assert.equal(
  context.isAuthorizedLiveAtcUrl(
    "https://www.liveatc.net/hlisten.php?icao=KMRY&mount=kmry",
  ),
  true,
);
assert.equal(
  context.isAuthorizedLiveAtcUrl(
    "https://www.liveatc.net/hlisten.php?mount=ksfo&icao=ksfo",
  ),
  false,
);
assert.equal(
  context.isAuthorizedLiveAtcUrl(
    "https://liveatc.example/hlisten.php?mount=kmry&icao=kmry",
  ),
  false,
);
assert.equal(
  context.isAuthorizedLiveAtcUrl(
    "http://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry",
  ),
  false,
);

assert.equal(await context.createNotification("success", {}), "created-id");

createImplementation = (_id, _options, callback) => {
  runtimeLastError = {message: "Unable to download all specified images."};
  callback(undefined);
  runtimeLastError = null;
};
await assert.rejects(
  context.createNotification("failure", {}),
  /Unable to download all specified images/,
);
assert.equal(loggedErrors.length, 1);

runtimeLastError = null;
createImplementation = (_id, _options, callback) => callback("created-id");
await context.rememberAndNotify({
  event_id: "test-event",
  test: true,
  status: "confirmed",
  confidence: 1,
  registration: "N123AB",
  spoken_callsign: "November one two three alpha bravo",
  aircraft_type: "Test aircraft",
  alternative_registrations: [],
  transcript_excerpt: "Local test event",
});
assert.equal(lastCreateOptions.iconUrl, "chrome-extension://test/icons/icon128.png");
assert.equal(lastCreateOptions.priority, 2);
assert.equal(lastCreateOptions.requireInteraction, true);
assert.equal(storageState.lastNotificationResult.ok, true);
assert.equal(storageState.lastNotificationResult.active, true);

storageState.mute = false;
storageState.minimumConfidence = 0.8;
lastCreateOptions = null;
await context.rememberAndNotify({
  event_id: "arrival-1",
  event_type: "arrival",
  confirmation_status: "pending",
  status: "confirmed",
  confidence: 0.95,
  registration: "N123AB",
  spoken_callsign: "November one two three alpha bravo",
  alternative_registrations: [],
  transcript_excerpt: "Taxi to Monterey Jet Center",
});
assert.equal(lastCreateOptions, null);
assert.equal(storageState.recentAlerts[0].confirmation_status, "pending");

await context.rememberAndNotify({
  event_id: "arrival-1",
  event_type: "arrival",
  confirmation_status: "confirmed",
  status: "confirmed",
  confidence: 0.95,
  registration: "N123AB",
  spoken_callsign: "November one two three alpha bravo",
  alternative_registrations: [],
  transcript_excerpt: "Taxi to Monterey Jet Center",
});
assert.equal(lastCreateOptions.title, "N123AB — Aircraft type unknown");
assert.match(lastCreateOptions.message, /Going to Monterey Jet Center/);

storageState.seenEventIds = [];
await context.rememberAndNotify({
  event_id: "possible-controller-route",
  event_type: "arrival",
  confirmation_status: "confirmed",
  status: "likely",
  confidence: 0.92,
  registration: "N830MG",
  spoken_callsign: "Unresolved aircraft",
  aircraft_type: "C650",
  destination: "Monterey Jet Center",
  alternative_registrations: [],
  transcript_excerpt: "Taxi Jet Center via Alpha Echo",
});
assert.match(lastCreateOptions.message, /Possible arrival/);

await context.rememberAndNotify({
  event_id: "correction-1",
  original_event_id: "arrival-1",
  event_type: "destination_correction",
  confirmation_status: "corrected",
  status: "confirmed",
  confidence: 0.95,
  registration: "N123AB",
  spoken_callsign: "November one two three alpha bravo",
  corrected_destination: "Del Monte Aviation",
  alternative_registrations: [],
  transcript_excerpt: "Actually Del Monte Aviation",
});
assert.equal(lastCreateOptions.title, "N123AB — Aircraft type unknown");
assert.equal(lastCreateOptions.priority, 2);
assert.equal(lastCreateOptions.requireInteraction, true);
assert.match(lastCreateOptions.message, /no longer expected at Monterey Jet Center/);
assert.equal(storageState.recentAlerts.filter((item) =>
  item.event_id === "arrival-1").length, 1);
assert.equal(storageState.recentAlerts.find((item) =>
  item.event_id === "arrival-1").confirmation_status, "corrected");

console.log("notification callback success and failure behavior verified");
