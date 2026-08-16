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

/** Inspect a GGUF's tool-calling capability without loading an engine. */
export function modelInfo(ggufPath) {
  const l = lib();
  return JSON.parse(takeString(l.modelInfo(ggufPath)) || '{}');
}

/**
 * Translate the request's camelCase options to the wire names the core reads.
 *
 * Only the fields this binding names are renamed; everything else (temperature,
 * max_tokens, seed, ...) is passed through untouched, so a new generation
 * parameter in the core needs no change here.
 *
 * `reuseCache` is deliberately only sent when the caller set it: the core's
 * default is true, and writing `reuse_cache: false` for an absent option would
 * turn the cache off for every existing caller.
 */
function toWire(request = {}) {
  const { jsonSchema, reuseCache, ...rest } = request;
  const wire = { ...rest };
  if (jsonSchema !== undefined) wire.json_schema = jsonSchema;
  if (reuseCache !== undefined) wire.reuse_cache = reuseCache;
  return wire;
}

/** A loaded model and its inference context. Call `close()` when done. */
export class Chat {
  /**
   * @param {string} ggufPath
   * @param {{nCtx?: number, nBatch?: number, nSeqMax?: number,
   *          onEvent?: (event: string) => void}} [options]
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
    if (options.nCtx !== undefined) config.n_ctx = options.nCtx;
    if (options.nBatch !== undefined) config.n_batch = options.nBatch;
    if (options.nSeqMax !== undefined) config.n_seq_max = options.nSeqMax;
    const configJson = Object.keys(config).length ? JSON.stringify(config) : null;

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
   * Generation parameters (temperature, top_k, top_p, min_p, max_tokens,
   * repeat_penalty, seed, stop) go inside `request`, so new ones need no change here.
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
   * `finish_reason: 'cancelled'`.
   *
   * @param {object} request
   * @param {(piece: string) => boolean|void} [onToken] pass to stream; the full response still returns
   */
  infer(request, onToken) {
    this._open();
    const payload = JSON.stringify(toWire(request));

    if (!onToken) {
      return check(takeString(this._lib.chatInfer(this._handle, payload)));
    }

    // Only an explicit `false` cancels. The usual callback -- one that writes a piece
    // and returns nothing -- yields undefined, and treating that falsy value as "stop"
    // would cancel every stream after its first token.
    const cb = koffi.register(
      (text) => (onToken(text ?? '') === false ? 1 : 0),
      koffi.pointer(this._lib.TokenCallback),
    );
    try {
      return check(takeString(this._lib.chatInferStream(this._handle, payload, cb, null)));
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
    const r = check(takeString(this._lib.countTokens(this._handle, JSON.stringify(toWire(request)))));
    return { tokens: r.tokens, nCtx: r.n_ctx };
  }

  // ---- LoRA ----

  _lora(op) {
    this._open();
    return check(takeString(this._lib.chatLora(this._handle, JSON.stringify(op))));
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
   *          onEvent?: (event: string) => void}} [options]
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

    const config = { n_batch: options.nBatch ?? 512 };
    if (options.pooling) config.pooling = options.pooling;
    if (options.nCtx) config.n_ctx = options.nCtx;

    const handle = this._lib.embedCreate(ggufPath, JSON.stringify(config), this._eventCb, null);
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
    const payload = JSON.stringify({ input, normalize });
    return check(takeString(this._lib.embed(this._handle, payload))).embeddings ?? [];
  }

  /**
   * Score documents against a query, best first.
   *
   * Each hit carries the document's ORIGINAL index, because results come back
   * reordered. Scores are raw model logits: comparable within one call, not across
   * models, and not probabilities. Requires pooling 'rank'.
   */
  rerank(query, documents, { topN } = {}) {
    this._open();
    const request = { query, documents };
    if (topN > 0) request.top_n = topN;
    return check(takeString(this._lib.rerank(this._handle, JSON.stringify(request)))).results ?? [];
  }

  /** Release the model and its context. Idempotent. */
  close() {
    if (!this._handle) return;
    this._lib.embedDestroy(this._handle);
    this._handle = null;
    koffi.unregister(this._eventCb);
  }
}
