# Spikes

Throwaway code and investigations that answer a **factual** question a decision depends on.

A spike exists to replace an argument with evidence. It has a README stating:

1. **Question** — the specific thing we do not know, phrased so it can be answered yes/no or
   with a number.
2. **Method** — what was actually done, reproducibly.
3. **Verdict** — the answer, and what it means for the decision waiting on it.

Rules:

- A spike is **never promoted to production code.** It is evidence; once the ADR or spec is
  written, it stops mattering. Keep it for the record, don't maintain it.
- Prefer a spike over a debate. If two people disagree about whether something is possible,
  that is a spike, not a meeting.
- A spike that fails is as valuable as one that succeeds — record the verdict either way.

| # | question | verdict |
|---|---|---|
| [0001](0001-bridge-portability/) | Is mochallama's `llamabridge` C surface bindable from non-Java FFIs as-is? | **Yes** — no Java coupling; extraction is packaging, not redesign |
