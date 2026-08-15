// Locating and loading the native bridge.
//
// koffi is used rather than node-gyp/N-API for the same reason purego is used in
// Go: no compiler on the user's machine, and no rebuild per Node version.

import koffi from 'koffi';
import { existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { arch, platform } from 'node:process';

/** The native bridge could not be located or loaded. */
export class NativeLibraryNotFoundError extends Error {
  constructor(searched, cause) {
    super(
      'modelnexus: native bridge not found or not loadable\n' +
        (cause ? `  cause: ${cause.message}\n` : '') +
        '  looked in:\n' +
        searched.map((p) => `    ${p}`).join('\n') +
        '\n  build it with core/build.sh, or set MODELNEXUS_LIB to the library or its directory',
    );
    this.name = 'NativeLibraryNotFoundError';
    this.searched = searched;
    this.cause = cause;
  }
}

/** The os-arch key used for the staged native directory layout. */
export function platformKey() {
  const os = platform === 'darwin' ? 'darwin' : platform === 'win32' ? 'windows' : 'linux';
  const cpu = arch === 'x64' ? 'x86_64' : arch === 'arm64' ? 'aarch64' : arch;
  return `${os}-${cpu}`;
}

function libFileName() {
  if (platform === 'darwin') return 'libllamabridge.dylib';
  if (platform === 'win32') return 'llamabridge.dll';
  return 'libllamabridge.so';
}

function candidates() {
  const name = libFileName();
  const out = [];

  const env = process.env.MODELNEXUS_LIB;
  if (env) {
    out.push(existsSync(env) && statSync(env).isDirectory() ? join(env, name) : env);
  }

  const here = dirname(fileURLToPath(import.meta.url));
  out.push(join(here, 'native', platformKey(), name));

  // Walk up to the repo's own build output, for working in the tree.
  let dir = here;
  for (let i = 0; i < 6; i++) {
    out.push(join(dir, 'core', 'dist', platformKey(), name));
    const parent = resolve(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  return out;
}

let bound = null;

/**
 * Bind the ABI once.
 *
 * Every function returning a string the caller owns is declared `void *`, never
 * `const char *`: koffi would decode a char* return into a JS string and drop the
 * pointer, making llb_string_free impossible to call and leaking every result.
 */
export function lib() {
  if (bound) return bound;

  const searched = candidates();
  let lastError = null;
  for (const path of searched) {
    if (!existsSync(path)) continue;
    let native;
    try {
      native = koffi.load(path);
    } catch (err) {
      // Present but unloadable is a different problem from absent.
      lastError = err;
      continue;
    }

    const StringCallback = koffi.proto('void StringCallback(const char *text, void *user)');

    bound = {
      koffi,
      StringCallback,
      version: native.func('const char *llb_version()'),
      modelInfo: native.func('void *llb_model_info(const char *path)'),
      chatCreate: native.func('void *llb_chat_create(const char *path, StringCallback *cb, void *user)'),
      chatInfer: native.func('void *llb_chat_infer(void *chat, const char *request)'),
      chatInferStream: native.func(
        'void *llb_chat_infer_stream(void *chat, const char *request, StringCallback *cb, void *user)',
      ),
      chatLora: native.func('void *llb_chat_lora(void *chat, const char *request)'),
      chatDestroy: native.func('void llb_chat_destroy(void *chat)'),
      embedCreate: native.func(
        'void *llb_embed_create(const char *path, const char *config, StringCallback *cb, void *user)',
      ),
      embed: native.func('void *llb_embed(void *embed, const char *request)'),
      rerank: native.func('void *llb_rerank(void *embed, const char *request)'),
      embedDestroy: native.func('void llb_embed_destroy(void *embed)'),
      stringFree: native.func('void llb_string_free(void *s)'),
    };
    return bound;
  }
  throw new NativeLibraryNotFoundError(searched, lastError);
}

/**
 * Copy a core-owned string and free the original.
 *
 * Copy and free together in one place, so the free cannot be forgotten at one of
 * the several call sites that need it.
 */
export function takeString(ptr) {
  if (!ptr) return '';
  const l = lib();
  try {
    return koffi.decode(ptr, 'char', -1);
  } finally {
    l.stringFree(ptr);
  }
}
