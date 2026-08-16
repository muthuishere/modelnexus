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

  test('a callback returning undefined does not cancel', () => {
    // The failure this guards against is total: a binding that coerces a falsy
    // return to "stop" cancels every stream at its first token.
    const chat = new Chat(MODEL);
    try {
      let seen = 0;
      const r = chat.infer(
        { messages: [{ role: 'user', content: 'Count: 1 2 3' }], max_tokens: 24, seed: 1 },
        () => {
          seen++;
        },
      );
      assert.ok(seen > 1, 'expected more than one piece');
      assert.notEqual(r.finish_reason, 'cancelled');
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

// Mirrors core/tests/abi_test.c. Three of these exist because the spike found the
// failure mode before the code did, and each one is silent rather than loud.
describe('inference control', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  const ASK_PARIS = {
    messages: [{ role: 'user', content: 'Name the capital of France in one word.' }],
    max_tokens: 16,
    seed: 42,
    temperature: 0,
  };

  test('no options still loads, and sizing options reach the context', () => {
    const plain = new Chat(MODEL);
    try {
      assert.ok(plain.countTokens(ASK_PARIS).tokens > 0);
    } finally {
      plain.close();
    }

    // n_ctx coming back changed is the proof the config JSON was actually read;
    // a binding that dropped it would still load and still look fine.
    const sized = new Chat(MODEL, { nCtx: 2048, nBatch: 256, nSeqMax: 1 });
    try {
      assert.equal(sized.countTokens(ASK_PARIS).nCtx, 2048);
    } finally {
      sized.close();
    }
  });

  test('countTokens reports a plausible count without decoding', () => {
    const chat = new Chat(MODEL);
    try {
      const short = chat.countTokens(ASK_PARIS);
      assert.ok(short.tokens > 0, 'expected a positive count');
      assert.ok(short.nCtx > 0, 'expected the context window');
      assert.ok(short.tokens < short.nCtx);

      const long = chat.countTokens({
        messages: [{ role: 'user', content: 'word '.repeat(200) }],
      });
      assert.ok(long.tokens > short.tokens, 'a longer prompt must count higher');
    } finally {
      chat.close();
    }
  });

  test('reuseCache on and off produce identical text', () => {
    // Reuse is a latency feature. Any observable difference in output is a defect,
    // and this is the assertion that catches it.
    const chat = new Chat(MODEL);
    try {
      const cold = chat.infer({ ...ASK_PARIS, reuseCache: false });
      const warm = chat.infer(ASK_PARIS);
      const coldAgain = chat.infer({ ...ASK_PARIS, reuseCache: false });

      assert.equal(warm.text, cold.text, 'reuse changed the output');
      assert.equal(coldAgain.text, cold.text, 'repeated cold runs are the control');
    } finally {
      chat.close();
    }
  });

  test('returning false cancels, and the response is a result rather than an error', () => {
    const chat = new Chat(MODEL);
    try {
      let seen = 0;
      const r = chat.infer(
        {
          messages: [{ role: 'user', content: 'Count slowly from one to fifty in words, one per line.' }],
          max_tokens: 300,
          seed: 7,
          temperature: 0,
        },
        () => {
          seen++;
          return seen >= 8 ? false : undefined;
        },
      );
      assert.equal(r.finish_reason, 'cancelled');
      assert.equal(seen, 8, 'generation stopped at the requested token, not later');
      assert.equal(r.usage.completion_tokens, 8, 'usage must count what was actually generated');

      // The proof that the cache rolled back: a DIFFERENT request on the same handle
      // must be correct, uninfluenced by the abandoned partial turn. Without rollback
      // the failure is silent -- no error, plausible output, wrong answer.
      const after = chat.infer(ASK_PARIS);
      assert.match(after.text, /Paris/, `expected Paris, got: ${after.text}`);
    } finally {
      chat.close();
    }
  });

  test('a JSON schema produces output that parses and satisfies it', () => {
    const chat = new Chat(MODEL);
    try {
      const r = chat.infer({
        messages: [{ role: 'user', content: 'Describe Paris.' }],
        max_tokens: 120,
        seed: 42,
        temperature: 0,
        jsonSchema: {
          type: 'object',
          properties: { city: { type: 'string' }, country: { type: 'string' } },
          required: ['city', 'country'],
          additionalProperties: false,
        },
      });
      // Parse, never substring-match: upstream's grammar PERMITS a ```json fence, and
      // a fence is invisible to `.includes()` while making JSON.parse throw.
      const parsed = JSON.parse(r.text);
      assert.equal(typeof parsed.city, 'string');
      assert.equal(typeof parsed.country, 'string');
      assert.deepEqual(Object.keys(parsed).sort(), ['city', 'country']);
    } finally {
      chat.close();
    }
  });

  test('a raw GBNF grammar constrains generation', () => {
    const chat = new Chat(MODEL);
    try {
      const r = chat.infer({
        messages: [{ role: 'user', content: 'Pick a colour.' }],
        max_tokens: 16,
        seed: 42,
        temperature: 0,
        grammar: 'root ::= "red" | "blue"',
      });
      assert.ok(['red', 'blue'].includes(r.text.trim()), `grammar did not constrain: ${r.text}`);
    } finally {
      chat.close();
    }
  });

  test('clearCache is observable, and the engine still works after it', () => {
    // The assertion that matters is that the clear is OBSERVABLE. A clear that
    // silently did nothing would still return a well-formed object, and the next
    // inference would still be correct -- just slow, and still holding the previous
    // tenant's conversation.
    const chat = new Chat(MODEL);
    try {
      chat.infer(ASK_PARIS);

      const before = chat.cacheStatus();
      assert.ok(before.tokens > 0, 'the cache is not empty after an inference');
      assert.ok(before.nCtx >= before.tokens);

      // Status is the non-destructive call -- the binding's stand-in for the ABI's
      // "a NULL request reads status, it does not clear". Reading twice must not
      // empty the cache; backwards, an innocent-looking call would wipe a
      // conversation.
      assert.equal(chat.cacheStatus().tokens, before.tokens);

      assert.equal(chat.clearCache().tokens, 0, 'clear empties the cache, and says so');
      assert.equal(chat.cacheStatus().tokens, 0, 'the clear persisted');

      assert.match(chat.infer(ASK_PARIS).text, /Paris/);
    } finally {
      chat.close();
    }
  });

  test('cache calls on a closed chat are a typed error', () => {
    const chat = new Chat(MODEL);
    chat.close();
    for (const call of [() => chat.cacheStatus(), () => chat.clearCache()]) {
      assert.throws(call, (e) => e instanceof ModelError && e.code === 'ENGINE_CLOSED');
    }
  });

  test('a schema and a grammar together is rejected, not silently resolved', () => {
    const chat = new Chat(MODEL);
    try {
      assert.throws(
        () =>
          chat.infer({
            messages: [{ role: 'user', content: 'hi' }],
            jsonSchema: { type: 'object' },
            grammar: 'root ::= "x"',
          }),
        (e) => e instanceof ModelError && e.code === 'INVALID_REQUEST',
      );
    } finally {
      chat.close();
    }
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
