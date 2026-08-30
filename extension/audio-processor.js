class MryPcmProcessor extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0) return true;

    for (let channel = 0; channel < output.length; channel += 1) {
      output[channel].set(input[channel] || input[0]);
    }

    const frames = input[0].length;
    const mono = new Float32Array(frames);
    for (let channel = 0; channel < input.length; channel += 1) {
      const samples = input[channel];
      for (let index = 0; index < frames; index += 1) {
        mono[index] += samples[index] / input.length;
      }
    }
    this.port.postMessage({samples: mono, sampleRate}, [mono.buffer]);
    return true;
  }
}

registerProcessor("mry-pcm-processor", MryPcmProcessor);
