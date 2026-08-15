# ADR-0003 — modelnexus and toolnexus do not depend on each other

- **Status:** accepted
- **Date:** 2026-08-15

## Context

modelnexus produces tool calls. toolnexus executes them. The obvious move is to have one
import the other so a user gets a working agent from a single dependency.

It is the wrong move in both directions:

- **toolnexus → modelnexus** would give a provider-agnostic tool library a native dependency
  and a hard link to one inference engine. It would also break toolnexus's Clojure port
  outright: that port is a single `.cljc` tree with **zero** reader conditionals whose
  `deps.edn` is guarded by `deps-purity-check.sh`, which fails on any transitive dependency
  beyond clojure + spec.alpha + core.specs.alpha + koine. Seven-language parity dies with it.
- **modelnexus → toolnexus** would put an agent loop inside an inference library, so anyone
  wanting raw generation would carry an orchestration framework.

## Decision

**Neither library depends on the other, in any direction, ever.** They meet at the OpenAI
wire shape and nowhere else.

The seam already sits in the right place: the core is built over llama.cpp's `common_chat`,
so it emits OpenAI-shaped `tool_calls` with `id`, `name` and `arguments`. That is **wire
format**, not orchestration. modelnexus proposes calls; toolnexus decides, executes, and
feeds results back as `role: "tool"` messages.

An application imports both and wires them in a few lines. That is the intended shape.

## Consequences

- toolnexus keeps its purity gate, its seven ports, and its provider neutrality.
- modelnexus is usable with no agent framework, and usable *by* any agent framework — Spring
  AI and LangChain4j sit above it just as well as toolnexus does.
- The convenience of one-import-gets-you-an-agent is deliberately given up. It is bought back
  with an example and a doc page, not a dependency edge.
- **The rule to hold:** the core must never grow a tool *executor* or a calling loop. The
  moment it does, there are two agent frameworks in the portfolio and this ADR is void.

## Alternatives rejected

- **A third glue package depending on both.** Plausible, but it becomes the thing everyone
  imports, so the coupling returns with an extra release artifact attached. Revisit only if
  the wiring example proves genuinely hard to follow.
