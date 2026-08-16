/**
 * The boundary between this binding's names and the core's.
 *
 * ONE RULE: the JS API is camelCase, the wire is snake_case, and every key crossing
 * the boundary is converted -- in both directions, recursively, with no exempt key.
 *
 * This is a general converter rather than a table of renamed keys on purpose. A table
 * is a list somebody has to remember to extend, it will not be extended for the next
 * key, and the result is a request object written half in each convention -- which is
 * precisely the defect this replaced.
 */

/** `maxTokens` -> `max_tokens`. */
function camelToSnakeKey(key) {
  return key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

/** `finish_reason` -> `finishReason`. */
function snakeToCamelKey(key) {
  return key.replace(/_([a-zA-Z0-9])/g, (_, c) => c.toUpperCase());
}

/**
 * Keys whose value is a JSON Schema the CALLER wrote. These are copied by reference
 * and never walked into.
 *
 * Two reasons, and both are about it not being our data. The property names inside
 * describe the output contract the model must satisfy, so renaming them rewrites what
 * was asked for. And property ORDER is load-bearing under grammar-constrained
 * decoding -- the model commits to fields in the order the grammar allows -- which is
 * how Go got a different answer for the same prompt once `encoding/json` sorted a
 * map's keys.
 *
 *   jsonSchema  -- the request's constrained-output schema
 *   parameters  -- tools[].function.parameters, the same thing once per tool
 */
const CALLER_SCHEMA_KEYS = new Set(['jsonSchema', 'parameters']);

/**
 * JS request -> wire JSON.
 *
 * A snake_case key is REJECTED rather than passed through. It cannot be anything but
 * a mistake: every core parameter is reachable by its camelCase spelling, including
 * ones this binding has never heard of, so there is no request a caller can only
 * express in snake_case. Ignoring it silently is the bug being removed -- `max_tokens`
 * used to marshal, reach the core, be ignored, and leave the call looking successful.
 */
export function toWire(value) {
  if (Array.isArray(value)) return value.map(toWire);
  if (value === null || typeof value !== 'object') return value;

  const out = {};
  for (const [key, v] of Object.entries(value)) {
    if (key.includes('_')) {
      throw new TypeError(
        `unknown option "${key}": the modelnexus JS API is camelCase — ` +
          `did you mean "${snakeToCamelKey(key)}"? snake_case is the wire spelling, ` +
          'and this binding does not accept it.',
      );
    }
    out[camelToSnakeKey(key)] = CALLER_SCHEMA_KEYS.has(key) ? v : toWire(v);
  }
  return out;
}

/** Wire JSON -> JS response. Everything this binding hands back is camelCase. */
export function fromWire(value) {
  if (Array.isArray(value)) {
    // An embedding is an array of numbers, and a number has no case. Rebuilding it
    // would copy every vector to change nothing.
    return value.some((v) => v !== null && typeof v === 'object') ? value.map(fromWire) : value;
  }
  if (value === null || typeof value !== 'object') return value;

  const out = {};
  for (const [key, v] of Object.entries(value)) out[snakeToCamelKey(key)] = fromWire(v);
  return out;
}
