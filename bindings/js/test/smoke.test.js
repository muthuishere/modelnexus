// Mirrors the Python and Go suites case for case. The four suites asserting the
// same behaviour -- and the same error codes -- is what keeps the bindings from
// drifting; that is the whole point of a shared C core.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { Chat, Embedder, ModelError, version, platformKey } from '../src/index.js';
// Reached directly rather than through the package entry point: the boundary is
// testable without a multi-gigabyte GGUF, and it is not public API.
import { toWire, fromWire } from '../src/wire.js';

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

describe('the camelCase boundary', () => {
  test('every request key is converted, not a named few', () => {
    // The defect this replaced renamed exactly jsonSchema and reuseCache and passed
    // the rest through, so a request was written half in each convention. A key this
    // binding has never heard of must convert too, or the next core parameter
    // reintroduces the split.
    assert.deepEqual(
      toWire({ maxTokens: 64, topK: 40, repeatPenalty: 1.1, reuseCache: false, mirostatTau: 5 }),
      { max_tokens: 64, top_k: 40, repeat_penalty: 1.1, reuse_cache: false, mirostat_tau: 5 },
    );
  });

  test('nested request objects are converted too', () => {
    assert.deepEqual(toWire({ messages: [{ role: 'tool', toolCallId: 'x' }] }), {
      messages: [{ role: 'tool', tool_call_id: 'x' }],
    });
  });

  test('a snake_case request option is REJECTED, never silently ignored', () => {
    // The decision, pinned: rejected. Every core parameter is reachable by its
    // camelCase spelling -- including ones this binding has never named -- so no
    // request exists that a caller can only write in snake_case, which makes a
    // snake_case key a mistake and nothing else. Ignoring it is the bug being
    // removed: `max_tokens: 16` used to marshal, reach the core, be ignored, and
    // leave the call looking like it had worked.
    assert.throws(
      () => toWire({ messages: [], max_tokens: 16 }),
      (e) => e instanceof TypeError && /maxTokens/.test(e.message),
    );
    // Nested, and inside an array, on the same rule.
    assert.throws(
      () => toWire({ messages: [{ role: 'tool', tool_call_id: 'x' }] }),
      (e) => e instanceof TypeError && /toolCallId/.test(e.message),
    );
  });

  test("a caller's JSON schema is passed through byte for byte", () => {
    // Its property names are the caller's contract with the model, and its property
    // ORDER is load-bearing under grammar-constrained decoding. Walking into it would
    // rewrite the question.
    const schema = {
      type: 'object',
      properties: { max_tokens: { type: 'integer' }, user_name: { type: 'string' } },
      required: ['max_tokens', 'user_name'],
      additionalProperties: false,
    };
    const wire = toWire({ jsonSchema: schema });
    assert.equal(wire.json_schema, schema, 'the schema must be the same object, not a copy');
    assert.deepEqual(Object.keys(wire.json_schema.properties), ['max_tokens', 'user_name']);

    // tools[].function.parameters is the same thing, once per tool.
    const params = { type: 'object', properties: { part_number: { type: 'string' } } };
    const tooled = toWire({
      tools: [{ type: 'function', function: { name: 'lookup', parameters: params } }],
    });
    assert.equal(tooled.tools[0].function.parameters, params);
  });

  test('every response key comes back camelCase', () => {
    assert.deepEqual(
      fromWire({
        type: 'assistant_text',
        finish_reason: 'stop',
        tool_calls: [{ id: '1', name: 'f', arguments: '{"a":1}' }],
        usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
      }),
      {
        type: 'assistant_text',
        finishReason: 'stop',
        toolCalls: [{ id: '1', name: 'f', arguments: '{"a":1}' }],
        usage: { promptTokens: 3, completionTokens: 4, totalTokens: 7 },
      },
    );
  });
});

describe('chat', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  test('infer returns text and usage', () => {
    const chat = new Chat(MODEL);
    try {
      const r = chat.infer({
        messages: [{ role: 'user', content: 'Reply with exactly: pong' }],
        maxTokens: 16,
        seed: 1,
      });
      assert.equal(r.type, 'assistant_text');
      assert.ok(r.text.length > 0);
      assert.ok(r.usage.totalTokens > 0);
    } finally {
      chat.close();
    }
  });

  test('streaming delivers pieces and the same final response', () => {
    const chat = new Chat(MODEL);
    try {
      const pieces = [];
      const r = chat.infer(
        { messages: [{ role: 'user', content: 'Count: 1 2 3' }], maxTokens: 24, seed: 1 },
        (p) => pieces.push(p),
      );
      assert.ok(pieces.length > 0, 'expected streamed pieces');
      assert.ok(r.usage.completionTokens > 0, 'expected usage on the streamed response');
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
      chat.infer({ messages: [{ role: 'user', content: 'hi' }], maxTokens: 8 });
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
        { messages: [{ role: 'user', content: 'Count: 1 2 3' }], maxTokens: 24, seed: 1 },
        () => {
          seen++;
        },
      );
      assert.ok(seen > 1, 'expected more than one piece');
      assert.notEqual(r.finishReason, 'cancelled');
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
    maxTokens: 16,
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
          maxTokens: 300,
          seed: 7,
          temperature: 0,
        },
        () => {
          seen++;
          return seen >= 8 ? false : undefined;
        },
      );
      assert.equal(r.finishReason, 'cancelled');
      assert.equal(seen, 8, 'generation stopped at the requested token, not later');
      assert.equal(r.usage.completionTokens, 8, 'usage must count what was actually generated');

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
        maxTokens: 120,
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
        maxTokens: 16,
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

describe('gpu layers', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  /** The config the CORE received, observed through its own event. */
  function createdConfig(options) {
    let seen = 'null';
    const c = new Chat(MODEL, {
      ...options,
      onEvent: (e) => {
        if (e.startsWith('create_config:')) seen = e.slice('create_config:'.length);
      },
    });
    try {
      return JSON.parse(seen);
    } finally {
      c.close();
    }
  }

  test('is absent unless asked for', () => {
    assert.equal(createdConfig({}), null);
  });

  test('0 is sent rather than read as unset', () => {
    // The assertion that matters. 0 means "CPU only" and is a deliberate request;
    // a truthy check would drop it and quietly hand the caller the GPU instead.
    assert.deepEqual(createdConfig({ nGpuLayers: 0 }), { n_gpu_layers: 0 });
  });
});

describe('embeddings', { skip: hasModel ? false : 'set MODELNEXUS_MODEL' }, () => {
  test('no options sends no config, rather than a default this binding invented', () => {
    // Observed at the core, not self-reported. This used to send {"n_batch":512}
    // unasked, which is a default the core owns pinned in a place nobody would look
    // when the core's own moved.
    const seen = [];
    const emb = new Embedder(MODEL, {
      onEvent: (e) => {
        if (e.startsWith('create_config:')) seen.push(e.slice('create_config:'.length));
      },
    });
    try {
      assert.deepEqual(seen, ['null']);
    } finally {
      emb.close();
    }
  });

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
