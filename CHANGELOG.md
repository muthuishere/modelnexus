# Changelog

All notable changes to modelnexus are recorded here. This file is what a user reads to learn
what changed — it is written with the change, not reconstructed from the git log at release
time.

## Unreleased

### Added
- Repository scaffolding: ADR log, spikes directory, OpenSpec workflow, and the house
  contribution rules in `CLAUDE.md`.
- ADR-0001 … ADR-0005 recording the decisions that define this repo: one runtime, the ABI as
  the product, independence from toolnexus, the two-tier native release model, and
  mochallama's demotion from owner to consumer of the core.
- Spike 0001 — a portability read of mochallama's `llamabridge`, concluding that the existing
  C surface carries no Java-specific coupling and can be extracted as-is.

### Not done yet
- No code. The C core has not been extracted, no binding exists, nothing is published.
- The ABI lacks LoRA adapter loading, embeddings, and reranking — the three capabilities that
  motivated this repo. Tracked in `openspec/changes/extract-llamabridge-core/`.
