/**
 * modelnexus -- local LLM inference, inside your own Node process.
 *
 *   import { Chat, Embedder } from '@muthuishere/modelnexus';
 *
 *   const chat = new Chat('model.gguf');
 *   const reply = chat.infer({ messages: [{ role: 'user', content: 'hello' }] });
 *   console.log(reply.text);
 *   chat.close();
 *
 * No server, no subprocess, no port. The model runs in this process.
 */

import koffi from 'koffi';
import { lib, takeString, platformKey, NativeLibraryNotFoundError } from './lib.js';
import { toWire, fromWire } from './wire.js';

export { platformKey, NativeLibraryNotFoundError };

/**
 * A failure reported by the core.
 *
 * `code` is stable and identical across every language binding; the message is for
 * humans.
 */
export class ModelError extends Error {
  constructor(code, message) {
    super(code ? `${code}: ${message}` : message);
    this.name = 'ModelError';
    this.code = code;
    this.modelMessage = message;
  }
}

function check(raw) {
  if (!raw) throw new ModelError('EMPTY_RESPONSE', 'the core returned nothing');
  const parsed = JSON.parse(raw);
  if (parsed.type === 'error') {
    const err = parsed.error ?? {};
    throw new ModelError(err.code ?? 'UNKNOWN', err.message ?? '');
  }
  return parsed;
}

/**
 * How much the inference engine is allowed to say.
 *
 * The bridge defaults to WARN rather than llama.cpp's own default: a library
 * embedded in someone else's process should be quiet unless asked.
 */
export const LogLevel = Object.freeze({
  NONE: 0,
  DEBUG: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
});

/**
 * Set how much the engine logs.
 *
 * Call before loading a model -- llama.cpp starts logging during load, so
 * afterwards is too late to silence it.
 */
export function setLogLevel(level) {
  lib().setLogLevel(level);
}

// The core retains the log callback until it is replaced, so the registration is
// held at module scope and released only when it is swapped out.
let logCallbackRef = null;

/** Route engine log output to a handler instead of stderr. Pass null to restore stderr. */
export function setLogHandler(handler) {
  const l = lib();
  if (logCallbackRef) {
    koffi.unregister(logCallbackRef);
    logCallbackRef = null;
  }
  if (!handler) {
    l.setLogCallback(null, null);
    return;
  }
  logCallbackRef = koffi.register(
    (level, text) => handler(level, text ?? ''),
    koffi.pointer(l.LogCallback),
  );
  l.setLogCallback(logCallbackRef, null);
}

/** Bridge version and the llama.cpp tag it was linked against. */
export function version() {
  return lib().version();
}

/**
 * Inspect a GGUF's tool-calling capability without loading an engine.
 *
 * @returns {{supportsTools: boolean, supportsToolCalls: boolean,
 *            hasToolUseTemplate: boolean, chatFormat: string|null, error: string|null}}
 */
export function modelInfo(ggufPath) {
  const l = lib();
  return fromWire(JSON.parse(takeString(l.modelInfo(ggufPath)) || '{}'));
}

/** Marshal a request and drop it on the wire, converting the answer back. */
function serialize(request) {
  // An absent option is absent on the wire: `undefined` survives toWire and
  // JSON.stringify drops it. That matters most for `reuseCache`, whose core default
  // is true -- writing `reuse_cache: false` for an unset option would turn the cache
  // off for every caller who never mentioned it.
  return JSON.stringify(toWire(request ?? {}));
}

/** A loaded model and its inference context. Call `close()` when done. */
export class Chat {
  /**
   * @param {string} ggufPath
   * @param {{nCtx?: number, nBatch?: number, nSeqMax?: number, nGpuLayers?: number,
   *          onEvent?: (event: string) => void}} [options]
   *
   * `nGpuLayers` is how many model layers are offloaded to the GPU. Unset means ALL
   * of them, which is the core's default and almost always what you want. Pass 0 for
   * CPU only -- a real setting rather than "unset", for a measurement that must be
   * reproducible across machines, or to leave the GPU for something else.
   */
  constructor(ggufPath, options = {}) {
    this._lib = lib();
    this._events = [];
    this._handle = null;

    // The core STORES this callback and calls it for the life of the handle, not
    // just during create. koffi.register pins the trampoline until unregister, so
    // it is kept on the instance and released in close() -- a callback that is
    // collected while native code still holds it crashes the process.
    this._eventCb = koffi.register((text) => {
      const value = text ?? '';
      this._events.push(value);
      options.onEvent?.(value);
    }, koffi.pointer(this._lib.StringCallback));

    // A null config is not the same as `{}`: the core reads NULL as "every default",
    // which is what this constructor did before the parameter existed. Callers who
    // pass no sizing option must land on that path byte for byte.
    const config = {};
    if (options.nCtx !== undefined) config.nCtx = options.nCtx;
    if (options.nBatch !== undefined) config.nBatch = options.nBatch;
    if (options.nSeqMax !== undefined) config.nSeqMax = options.nSeqMax;
    // `!== undefined`, never a truthy test: 0 is a legitimate nGpuLayers meaning
    // "CPU only", and dropping it would silently hand the caller the GPU instead.
    // The name is nGpuLayers, not nGPULayers, because toWire lowercases each capital
    // on its own -- nGPULayers would cross the wire as `n_g_p_u_layers`.
    if (options.nGpuLayers !== undefined) config.nGpuLayers = options.nGpuLayers;
    const configJson = Object.keys(config).length ? serialize(config) : null;

    const handle = this._lib.chatCreate(ggufPath, configJson, this._eventCb, null);
    if (!handle) {
      koffi.unregister(this._eventCb);
      // NULL is the one place the core signals failure with a null pointer; the
      // reason arrives through the event callback instead.
      if (this._events.some((e) => e.includes('tools_unsupported'))) {
        throw new ModelError('MODEL_NOT_TOOL_CAPABLE', `${ggufPath} has no tool-calling chat template`);
      }
      const detail = this._events.join('; ') || 'unknown reason';
      throw new ModelError('MODEL_LOAD_FAILED', `could not load ${ggufPath} (${detail})`);
    }
    this._handle = handle;
  }

  _open() {
    if (!this._handle) throw new ModelError('ENGINE_CLOSED', 'this Chat has already been closed');
  }

  /**
   * Run one turn.
   *
   * Generation parameters (temperature, topK, topP, minP, maxTokens, repeatPenalty,
   * seed, stop) go inside `request`, so new ones need no change here: the boundary
   * converts camelCase to the core's snake_case whether or not this binding has
   * heard of the key.
   *
   * Constrained output: `jsonSchema` (an object) guarantees the reply parses as JSON
   * and satisfies the schema, or `grammar` (raw GBNF). Supplying both is an
   * INVALID_REQUEST error from the core rather than a precedence rule.
   *
   * `reuseCache: false` makes the call provably independent of the ones before it.
   * The default -- reuse -- is a latency property only; output is identical.
   *
   * Returning `false` from `onToken` stops generation. That is not an error: the
   * response comes back complete, with the text produced so far and
   * `finishReason: 'cancelled'`.
   *
   * @param {object} request
   * @param {(piece: string) => boolean|void} [onToken] pass to stream; the full response still returns
   * @returns {{type: string, text: string, toolCalls: object[], finishReason: string,
   *            usage: {promptTokens: number, completionTokens: number, totalTokens: number}}}
   */
  infer(request, onToken) {
    this._open();
    const payload = serialize(request);

    if (!onToken) {
      return fromWire(check(takeString(this._lib.chatInfer(this._handle, payload))));
    }

    // Only an explicit `false` cancels. The usual callback -- one that writes a piece
    // and returns nothing -- yields undefined, and treating that falsy value as "stop"
    // would cancel every stream after its first token.
    const cb = koffi.register(
      (text) => (onToken(text ?? '') === false ? 1 : 0),
      koffi.pointer(this._lib.TokenCallback),
    );
    try {
      return fromWire(check(takeString(this._lib.chatInferStream(this._handle, payload, cb, null))));
    } finally {
      // The token callback is only needed for the duration of the call, unlike the
      // event callback -- release it rather than leaking a trampoline per inference.
      koffi.unregister(cb);
    }
  }

  /**
   * How many tokens a request's messages will occupy, and the context window they
   * have to fit in. Decodes nothing and does not disturb the KV cache.
   *
   * Lives in the core rather than here because counting needs the model's vocabulary
   * AND its parsed chat template, and a binding holds neither.
   *
   * @param {object} request the same `messages` (and optional `tools`) shape as infer
   * @returns {{tokens: number, nCtx: number}}
   */
  countTokens(request) {
    this._open();
    const r = fromWire(check(takeString(this._lib.countTokens(this._handle, serialize(request)))));
    return { tokens: r.tokens, nCtx: r.nCtx };
  }

  // ---- KV cache ----

  _cache(op) {
    this._open();
    const r = fromWire(check(takeString(this._lib.chatCache(this._handle, serialize({ op })))));
    return { tokens: r.tokens, nCtx: r.nCtx };
  }

  /**
   * What the engine's KV cache currently holds. Changes nothing.
   *
   * @returns {{tokens: number, nCtx: number}}
   */
  cacheStatus() {
    return this._cache('status');
  }

  /**
   * Drop the KV cache, freeing its memory and forgetting the sequence. Returns the
   * state AFTERWARDS -- always zero tokens, so a caller can assert the release
   * happened rather than trust that it did.
   *
   * Prefix reuse is right for a conversation that appends and wrong when a chat moves
   * to unrelated work: the old conversation keeps occupying context memory, and two
   * tenants sharing a handle would share a cache. Passing `reuseCache: false` on the
   * next inference also clears, but only as a side effect of doing work -- no help
   * when the point is to release memory now, or to prove the cache is empty before
   * handing the handle on.
   *
   * @returns {{tokens: number, nCtx: number}}
   */
  clearCache() {
    return this._cache('clear');
  }

  // ---- LoRA ----

  _lora(op) {
    this._open();
    return fromWire(check(takeString(this._lib.chatLora(this._handle, serialize(op)))));
  }

  /**
   * Load a LoRA adapter and apply it. Returns its id.
   *
   * Adapters change *behaviour* -- format, tone, tool-call reliability -- not
   * knowledge. For facts, retrieve.
   */
  loadLora(path, scale = 1.0) {
    return this._lora({ op: 'load', path, scale }).id;
  }

  /** Change an adapter's scale. Takes effect on the next inference. */
  setLoraScale(id, scale) {
    this._lora({ op: 'set', id, scale });
  }

  /** Unload one adapter and reapply the rest. */
  removeLora(id) {
    this._lora({ op: 'remove', id });
  }

  /** Unload every adapter, returning the model to its base behaviour. */
  clearLoras() {
    this._lora({ op: 'clear' });
  }

  /** The adapters currently applied, in order. */
  loras() {
    return this._lora({ op: 'list' }).adapters ?? [];
  }

  /** Release the model and its context. Idempotent. */
  close() {
    if (!this._handle) return;
    this._lib.chatDestroy(this._handle);
    this._handle = null;
    koffi.unregister(this._eventCb);
  }
}

/**
 * A model loaded for embedding or reranking.
 *
 * Separate from Chat because embedding needs a context built with embeddings
 * enabled and a pooling strategy fixed at creation, and reranking needs
 * pooling 'rank' specifically -- neither can be switched afterwards.
 */
export class Embedder {
  /**
   * @param {string} ggufPath
   * @param {{pooling?: 'mean'|'cls'|'last'|'rank'|'none', nCtx?: number, nBatch?: number,
   *          nGpuLayers?: number, onEvent?: (event: string) => void}} [options]
   *
   * `nGpuLayers` behaves exactly as it does on Chat: unset means all layers, and 0
   * is CPU only -- a deliberate setting, not "unset".
   */
  constructor(ggufPath, options = {}) {
    this._lib = lib();
    this._events = [];
    this._handle = null;

    this._eventCb = koffi.register((text) => {
      const value = text ?? '';
      this._events.push(value);
      options.onEvent?.(value);
    }, koffi.pointer(this._lib.StringCallback));

    // Only what the caller set, exactly as Chat does above. A binding that restates a
    // default -- this one used to send n_batch: 512 unasked -- pins the old value the
    // day the core moves its own, and does it silently. The core owns every default.
    const config = {};
    if (options.pooling !== undefined) config.pooling = options.pooling;
    if (options.nCtx !== undefined) config.nCtx = options.nCtx;
    if (options.nBatch !== undefined) config.nBatch = options.nBatch;
    // Same rule as Chat: 0 means CPU only, not unset.
    if (options.nGpuLayers !== undefined) config.nGpuLayers = options.nGpuLayers;
    const configJson = Object.keys(config).length ? serialize(config) : null;

    const handle = this._lib.embedCreate(ggufPath, configJson, this._eventCb, null);
    if (!handle) {
      koffi.unregister(this._eventCb);
      const detail = this._events.join('; ') || 'unknown reason';
      throw new ModelError('MODEL_LOAD_FAILED', `could not load ${ggufPath} (${detail})`);
    }
    this._handle = handle;
  }

  _open() {
    if (!this._handle) throw new ModelError('ENGINE_CLOSED', 'this Embedder has already been closed');
  }

  /**
   * Embed one string or many. Returns one vector per input, in input order.
   *
   * L2-normalized by default, which is what makes a dot product a cosine similarity.
   */
  embed(texts, { normalize = true } = {}) {
    this._open();
    const input = Array.isArray(texts) ? texts : [texts];
    const payload = serialize({ input, normalize });
    return fromWire(check(takeString(this._lib.embed(this._handle, payload)))).embeddings ?? [];
  }

  /**
   * Score documents against a query, best first.
   *
   * Each hit carries the document's ORIGINAL index, because results come back
   * reordered. Scores are raw model logits: comparable within one call, not across
   * models, and not probabilities. Requires pooling 'rank'.
   *
   * @returns {{index: number, score: number}[]}
   */
  rerank(query, documents, { topN } = {}) {
    this._open();
    const request = { query, documents };
    if (topN > 0) request.topN = topN;
    return fromWire(check(takeString(this._lib.rerank(this._handle, serialize(request))))).results ?? [];
  }

  /** Release the model and its context. Idempotent. */
  close() {
    if (!this._handle) return;
    this._lib.embedDestroy(this._handle);
    this._handle = null;
    koffi.unregister(this._eventCb);
  }
}
