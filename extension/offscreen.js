const TARGET_SAMPLE_RATE = 16000;
const SEND_SAMPLES = 1600;

let mediaStream;
let audioContext;
let processorNode;
let audioSocket;
let pcmQueue = [];
let queuedSamples = 0;
let backendReady = false;
let resamplePosition = 0;
let previousSample;

function report(status, detail = "", transcript = "") {
  chrome.runtime.sendMessage({
    type: "liveatc-capture-status",
    target: "background",
    status,
    detail,
    transcript
  });
}

function resample(input, inputRate) {
  if (inputRate === TARGET_SAMPLE_RATE) return input;
  const ratio = inputRate / TARGET_SAMPLE_RATE;
  const combined = previousSample === undefined ? input :
    Float32Array.from([previousSample, ...input]);
  const output = [];
  while (resamplePosition < combined.length - 1) {
    const left = Math.floor(resamplePosition);
    const fraction = resamplePosition - left;
    output.push(combined[left] + (combined[left + 1] - combined[left]) * fraction);
    resamplePosition += ratio;
  }
  resamplePosition -= combined.length - 1;
  previousSample = combined[combined.length - 1];
  return Float32Array.from(output);
}

function encodePcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
  }
  return buffer;
}

function sendAvailablePcm() {
  while (queuedSamples >= SEND_SAMPLES && audioSocket?.readyState === WebSocket.OPEN) {
    const outgoing = new Float32Array(SEND_SAMPLES);
    let offset = 0;
    while (offset < SEND_SAMPLES) {
      const first = pcmQueue[0];
      const count = Math.min(first.length, SEND_SAMPLES - offset);
      outgoing.set(first.subarray(0, count), offset);
      offset += count;
      queuedSamples -= count;
      if (count === first.length) pcmQueue.shift();
      else pcmQueue[0] = first.subarray(count);
    }
    audioSocket.send(encodePcm16(outgoing));
  }
}

async function stopCapture(detail = "Capture stopped.") {
  if (processorNode) processorNode.disconnect();
  processorNode = null;
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  if (audioContext) await audioContext.close();
  audioContext = null;
  if (audioSocket) {
    audioSocket.onclose = null;
    audioSocket.close();
  }
  audioSocket = null;
  backendReady = false;
  pcmQueue = [];
  queuedSamples = 0;
  resamplePosition = 0;
  previousSample = undefined;
  report("stopped", detail);
}

async function startCapture(data) {
  await stopCapture("Preparing authorized KMRY capture.");
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: "tab",
        chromeMediaSourceId: data.streamId
      }
    },
    video: false
  });
  const track = mediaStream.getAudioTracks()[0];
  if (!track) throw new Error("The selected LiveATC tab has no capturable audio track.");
  track.onended = () => stopCapture("The captured tab audio ended.");

  audioContext = new AudioContext({sampleRate: TARGET_SAMPLE_RATE});
  await audioContext.audioWorklet.addModule(chrome.runtime.getURL("audio-processor.js"));
  const source = audioContext.createMediaStreamSource(mediaStream);
  processorNode = new AudioWorkletNode(audioContext, "mry-pcm-processor");
  processorNode.port.onmessage = ({data: packet}) => {
    if (!backendReady) return;
    const samples = resample(packet.samples, packet.sampleRate);
    if (samples.length === 0) return;
    pcmQueue.push(samples);
    queuedSamples += samples.length;
    sendAvailablePcm();
  };
  source.connect(processorNode);
  processorNode.connect(audioContext.destination);

  const websocketUrl = data.backendUrl.replace(/^http/, "ws").replace(/\/$/, "") +
    "/ws/audio?token=" + encodeURIComponent(data.pairingToken);
  audioSocket = new WebSocket(websocketUrl);
  audioSocket.binaryType = "arraybuffer";
  audioSocket.onopen = () => {
    report("connected", "Capturing authorized KMRY player audio; waiting for local model.");
    sendAvailablePcm();
  };
  audioSocket.onmessage = (message) => {
    const data = JSON.parse(message.data);
    if (data.type === "status") {
      backendReady = data.status === "monitoring";
      report(data.status, `Backend audio status: ${data.status}.`);
    }
    if (data.type === "transcript") {
      report("monitoring", data.alert_created ? "Alert created." : "Transmission processed.", data.text);
    }
    if (data.type === "error") report("error", data.message);
  };
  audioSocket.onerror = () => report("error", "Could not connect to backend audio input.");
  audioSocket.onclose = (event) => {
    if (event.code !== 1000) report("error", `Backend audio connection closed (${event.code}).`);
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.target !== "offscreen") return false;
  if (message.type === "start-liveatc-capture") {
    startCapture(message.data)
      .then(() => sendResponse({ok: true, message: "Authorized KMRY capture started."}))
      .catch(async (error) => {
        console.error("MRY offscreen capture failed:", error);
        await stopCapture(error.message);
        report("error", error.message);
        sendResponse({ok: false, message: error.message});
      });
    return true;
  }
  if (message.type === "stop-liveatc-capture") {
    stopCapture().then(() => sendResponse({ok: true}));
    return true;
  }
  return false;
});
