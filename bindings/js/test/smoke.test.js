// Mirrors the Python and Go suites case for case. The four suites asserting the
// same behaviour -- and the same error codes -- is what keeps the bindings from
// drifting; that is the whole point of a shared C core.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { Chat, Embedder, ModelError, version, platformKey } from '../src/index.js';

// Skip rather than fail without a model: the binding is worth testing on a machine
// with no multi-gigabyte GGUF sitting around.
const MODEL = process.env.MODELNEXUS_MODEL;
const RERANKER = process.env.MODELNEXUS_RERANKER;
const hasModel = Boolean(MODEL && existsSync(MODEL));
const hasReranker = Boolean(RERANKER && existsSync(RERANKER));

test('version identifies the bridge and the engine', () => {
  const v = version();
  assert.match(v, /llamabridge/);
  assert.match(v, /llama\.cpp/);
});

test('platformKey looks like os-arch', () => {
  assert.match(platformKey(), /^(darwin|linux|windows)-(x86_64|aarch64)$/);
});

describe('chat', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  test('infer returns text and usage', () => {
    const chat = new Chat(MODEL);
    try {
      const r = chat.infer({
        messages: [{ role: 'user', content: 'Reply with exactly: pong' }],
        max_tokens: 16,
        seed: 1,
      });
      assert.equal(r.type, 'assistant_text');
      assert.ok(r.text.length > 0);
      assert.ok(r.usage.total_tokens > 0);
    } finally {
      chat.close();
    }
  });

  test('streaming delivers pieces and the same final response', () => {
    const chat = new Chat(MODEL);
    try {
      const pieces = [];
      const r = chat.infer(
        { messages: [{ role: 'user', content: 'Count: 1 2 3' }], max_tokens: 24, seed: 1 },
        (p) => pieces.push(p),
      );
      assert.ok(pieces.length > 0, 'expected streamed pieces');
      assert.ok(r.usage.completion_tokens > 0, 'expected usage on the streamed response');
    } finally {
      chat.close();
    }
  });

  test('the event callback survives past construction', () => {
    // The core keeps the event callback for the life of the handle; a binding that
    // lets the trampoline be collected crashes the process on the next emit.
    const events = [];
    const chat = new Chat(MODEL, { onEvent: (e) => events.push(e) });
    try {
      chat.infer({ messages: [{ role: 'user', content: 'hi' }], max_tokens: 8 });
      assert.ok(events.length > 0);
    } finally {
      chat.close();
    }
  });

  test('LoRA starts empty and clearing an empty set is a no-op', () => {
    const chat = new Chat(MODEL);
    try {
      assert.deepEqual(chat.loras(), []);
      chat.clearLoras();
    } finally {
      chat.close();
    }
  });

  test('LoRA errors carry the shared codes', () => {
    const chat = new Chat(MODEL);
    try {
      assert.throws(() => chat.loadLora('/definitely/not/here.gguf'), (e) => e.code === 'LORA_LOAD_FAILED');
      assert.throws(() => chat.setLoraScale(99, 0.5), (e) => e.code === 'LORA_NOT_FOUND');
    } finally {
      chat.close();
    }
  });

  test('use after close is rejected', () => {
    const chat = new Chat(MODEL);
    chat.close();
    chat.close(); // idempotent
    assert.throws(() => chat.infer({ messages: [] }), (e) => e.code === 'ENGINE_CLOSED');
  });
});

test('a missing model is a typed error', () => {
  assert.throws(
    () => new Chat('/definitely/not/here.gguf'),
    (e) => e instanceof ModelError && e.code === 'MODEL_LOAD_FAILED',
  );
});

describe('embeddings', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  test('embed returns unit vectors, one per input', () => {
    const emb = new Embedder(MODEL, { pooling: 'mean', nCtx: 512 });
    try {
      const v = emb.embed(['hello world', 'goodbye world']);
      assert.equal(v.length, 2);
      assert.equal(v[0].length, v[1].length);
      // Normalization is what makes a dot product a cosine similarity; if it drifts,
      // every downstream similarity silently changes meaning.
      const norm = Math.sqrt(v[0].reduce((s, x) => s + x * x, 0));
      assert.ok(Math.abs(norm - 1) < 1e-3, `L2 norm was ${norm}`);
    } finally {
      emb.close();
    }
  });

  test('rerank refuses an engine without rank pooling', () => {
    const emb = new Embedder(MODEL, { pooling: 'mean', nCtx: 512 });
    try {
      assert.throws(() => emb.rerank('q', ['a']), (e) => e.code === 'POOLING_NOT_RANK');
    } finally {
      emb.close();
    }
  });
});

test('rerank ranks semantically', { skip: hasReranker ? false : 'set MODELNEXUS_RERANKER' }, () => {
  const rr = new Embedder(RERANKER, { pooling: 'rank', nCtx: 512 });
  try {
    const docs = [
      'Berlin is the capital and largest city of Germany.',
      "Paris has been France's capital since the 10th century.",
      'Bananas are a good source of potassium.',
    ];
    const hits = rr.rerank('What is the capital of France?', docs);
    assert.equal(hits.length, 3);
    assert.equal(hits[0].index, 1, 'the Paris document must win');
    for (let i = 1; i < hits.length; i++) {
      assert.ok(hits[i].score <= hits[i - 1].score, 'results must be sorted best-first');
    }
    assert.equal(rr.rerank('What is the capital of France?', docs, { topN: 1 }).length, 1);
  } finally {
    rr.close();
  }
});
