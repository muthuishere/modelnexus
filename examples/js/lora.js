// lora -- apply a LoRA adapter to a live engine, then take it off again.
//
// Adapters load against the model already in memory: no reload, no second copy of the
// weights, and several can be active at once with independent scales. They change
// *behaviour* — tone, output format, tool-call reliability — not knowledge. For facts,
// retrieve.
//
// The adapter and the base model are a matched PAIR: an adapter is built for one
// architecture and one tensor layout, and will not load against an arbitrary GGUF.
// Hence two env vars rather than reusing MODELNEXUS_MODEL.
//
//   MODELNEXUS_LORA_BASE=/path/to/base.gguf MODELNEXUS_LORA=/path/to/adapter.gguf node lora.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const base = process.env.MODELNEXUS_LORA_BASE;
const adapter = process.env.MODELNEXUS_LORA;
if (!base || !adapter) {
  console.error('set MODELNEXUS_LORA_BASE and MODELNEXUS_LORA to a matched base/adapter pair');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

// The adapter used to develop this example removes the base model's refusal behaviour,
// so a prompt the base declines is the one place the difference is legible.
const prompt = 'Say something rude about the weather in one sentence.';

// Scale is a dial, not a switch. This adapter is f16 against a q4 base, and at 1.0 it
// overwhelms the model — output degenerates into fragments. 0.25 shifts behaviour and
// keeps the model coherent. Any adapter you did not train yourself deserves this sweep.
const scale = 0.25;

const chat = new Chat(base);
let before;
let during;
let after;
try {
  // Temperature 0 and a fixed seed, so the only thing that can move the output between
  // the three calls below is the adapter.
  const ask = () =>
    chat.infer({
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.0,
      seed: 3,
      max_tokens: 60,
      // Each call must be provably independent, or the previous call's KV prefix —
      // computed under a different adapter set — could be reused underneath it.
      reuseCache: false,
    }).text;

  console.log('prompt:', prompt);
  console.log();

  before = ask();
  console.log('--- base model ---');
  console.log(before);
  console.log();

  const id = chat.loadLora(adapter, scale);
  const applied = chat.loras();
  console.log(`--- adapter ${id} applied at scale ${applied[0].scale.toFixed(2)} ---`);
  during = ask();
  console.log(during);
  console.log();

  // clearLoras unloads every adapter and reapplies nothing, so the engine is back to
  // the weights it loaded from disk.
  chat.clearLoras();
  after = ask();
  console.log('--- adapter cleared ---');
  console.log(after);
  console.log();
} finally {
  chat.close();
}

if (before !== after) {
  // This is the check worth failing on: removing an adapter must restore the base model
  // exactly, or the engine is carrying state it should not.
  console.error('clearing the adapter did not restore the base model’s output');
  process.exit(1);
}

console.log('clearing restored the base output byte for byte.');
console.log(
  before === during
    ? 'this adapter did not change the answer to this particular prompt.'
    : 'the adapter changed the answer; the base model was restored by clearing it.',
);
