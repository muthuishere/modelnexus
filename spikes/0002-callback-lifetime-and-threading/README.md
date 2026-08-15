# Spike 0002 — callback lifetime across the FFI boundary

- **Date:** 2026-08-15
- **Verdict:** **Bindable everywhere, but the core RETAINS the event callback** for
  the life of the engine. A binding that lets the trampoline be collected after
  `create` returns crashes the process. Verified by crashing it.

## Question

`llb_event_cb` and `llb_token_cb` are plain C function pointers. Spike 0001 flagged
them as the one part of the surface whose portability could not be settled by reading
the header. Two unknowns:

1. Can each runtime hand a callable to native code at all?
2. What is the required **lifetime** of that callable?

## Method

Built the core (`core/build.sh`, prebuilt mode, llama.cpp b9371) and drove it from
Python/ctypes against a real 1.5B GGUF, then repeated the same sequence from
Go/purego with `CGO_ENABLED=0`.

## Findings

### 1. The core stores the event callback — this is the important one

`llamabridge.cpp:540` does:

```c
chat->event_cb  = event_cb;
chat->user_data = user_data;
```

and `emit()` (line 78) calls it later, throughout the engine's life — not just
during `llb_chat_create`. **The header does not say this.** It documents `event_cb`
as a "progress callback" for load, which reads as call-scoped.

Consequence, observed for real: a first Python binding that held the trampoline in a
local inside `__init__` segfaulted (exit 139) partway through the first inference —
the object was collected as soon as the constructor returned, and the next `emit()`
jumped into freed memory. Isolated tests passed, because in those the callable
happened to stay alive at module scope. **This is a bug that hides in exactly the
tests you would write first.**

The token callback, by contrast, is used only for the duration of
`llb_chat_infer_stream` and needs no lifetime beyond the call.

### 2. Both runtimes can take the callback

| runtime | mechanism | result |
|---|---|---|
| Python | `ctypes.CFUNCTYPE` | works; must be held on the instance |
| Go | `purego.NewCallback` | works with `CGO_ENABLED=0` |

No threading problem appeared: callbacks arrived on the calling thread in both
runtimes, and no GIL or scheduler interaction was needed. **Not proven for Java,
C# or JS** — the same check must be repeated when those bindings are written.

### 3. purego's callback allocation is capped and permanent

`purego.NewCallback` never releases a trampoline and the platform limits how many can
exist. One per `Chat` would leak and eventually fail outright in a long-lived process.

The Go binding therefore creates **exactly one trampoline per callback type, ever**,
and routes per-instance state through `user_data` carrying an integer key into a
registry — an integer, not a Go pointer, because native code must not hold a pointer
the garbage collector can move.

## What this means

Two things become contract, not implementation detail:

1. **Every binding must keep the event callback alive for the lifetime of the engine
   handle**, and release it at destroy. This belongs in the spec, because it is a
   property of the *core*, and every future binding will otherwise rediscover it as a
   crash.
2. **The header should say so.** `llb_chat_create`'s doc comment describes the event
   callback without stating its retention. That is a documentation defect in the ABI
   and should be fixed in the core, not worked around per binding.

## Also surfaced

The core has **no log control**. llama.cpp writes verbosely to stderr and the ABI has
no `llb_set_log_*` entry point, so any host application inherits several hundred lines
of engine chatter it cannot switch off. That is a real gap for a library meant to be
embedded, and it needs an ABI addition.

## Not covered

- Java/Panama, C#/P-Invoke, JS/koffi — callbacks unverified.
- Callbacks invoked from a genuinely non-caller thread. Nothing in the current core
  does that, so it stayed untested; it would matter if generation ever moves to a
  worker thread.
