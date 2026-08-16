# model-core

## ADDED Requirements

### Requirement: Caller-supplied output constraints

The core SHALL accept an optional JSON schema or an optional raw GBNF grammar on an inference
request, and SHALL constrain generation to it. The core SHALL map each source to its own
grammar type — a schema to the output-format type and a raw grammar to the user type — because
the types differ in whether the generation prompt is prefilled into the grammar sampler.

#### Scenario: A request supplies a JSON schema

- **WHEN** an inference request carries a `json_schema`
- **THEN** generation is constrained to that schema
- **AND** the returned content parses as JSON and satisfies the schema
- **AND** no exported C symbol changes signature

#### Scenario: A request supplies a raw GBNF grammar

- **WHEN** an inference request carries a `grammar`
- **THEN** generation is constrained to that grammar
- **AND** the grammar is NOT prefilled with the generation prompt

#### Scenario: A request supplies both a schema and a grammar

- **WHEN** an inference request carries both `json_schema` and `grammar`
- **THEN** the request is rejected with a stable error code
- **AND** no generation occurs

#### Scenario: Constrained output arrives wrapped in a markdown fence

- **WHEN** the constraining grammar permits a fenced code block and the model emits one
- **THEN** the core strips the fence before returning
- **AND** the caller receives content that parses as JSON without further processing

### Requirement: Token counting without inference

The core SHALL report the token count of a message list without creating an inference context
and without decoding, so that a caller can budget context before committing to a call.

#### Scenario: A caller counts a conversation

- **WHEN** a caller submits a message list for counting
- **THEN** the core applies the model's chat template, tokenizes, and returns the token count
- **AND** it returns the model's context size alongside it
- **AND** no generation occurs and no KV cache state changes

#### Scenario: A binding attempts to count tokens itself

- **WHEN** a binding proposes computing a token count locally
- **THEN** the proposal is rejected, because counting requires the model's vocabulary and its
  parsed chat template, neither of which a binding holds

### Requirement: Engine creation accepts configuration

`llb_chat_create` SHALL accept a configuration JSON parameter carrying at least context size,
batch size, and maximum sequence count, mirroring `llb_embed_create`. A NULL configuration
SHALL behave exactly as the previous signature did.

#### Scenario: A caller sets a context size

- **WHEN** an engine is created with a configuration specifying a context size
- **THEN** the engine uses it

#### Scenario: A caller passes no configuration

- **WHEN** an engine is created with a NULL configuration
- **THEN** the engine's behaviour is identical to that of the previous signature

### Requirement: Streaming token delivery is cancellable

The core SHALL deliver generated tokens to a caller-supplied callback as they are produced, and
that callback SHALL be able to stop generation. The callback returns an integer; a non-zero
return SHALL end generation.

Cancellation SHALL produce a complete, well-formed response carrying the partial text, a
`finish_reason` of `cancelled`, and honest usage counts — a cancelled generation is a result,
not an error, and it consumed real work.

Cancellation SHALL roll the KV cache back to the prompt boundary and truncate the retained
token sequence to match, so that a later prefix match cannot extend a truncated turn as though
it were complete.

#### Scenario: A consumer stops a stream mid-generation

- **WHEN** the token callback returns non-zero
- **THEN** generation stops without decoding further tokens
- **AND** a complete response is returned with `"finish_reason": "cancelled"`
- **AND** the usage counts reflect the tokens actually generated

#### Scenario: A consumer lets generation finish

- **WHEN** the token callback returns zero for every token
- **THEN** generation runs to its natural end
- **AND** the response is identical to the same request made without a callback

#### Scenario: A new request follows a cancelled one

- **WHEN** a generation is cancelled and a different request is then made on the same engine
- **THEN** the cache has been rolled back to the prompt boundary
- **AND** the new response is correct, with no influence from the abandoned partial output

### Requirement: Inference reuses prior work

The core SHALL retain the token sequence resident in the KV cache between calls on an engine,
and SHALL reuse the longest common prefix between that sequence and a new prompt, decoding only
the divergent tail. Reuse SHALL be the default. A request field SHALL disable it for callers who
require each call to be provably independent.

Reuse SHALL NOT change generated output. It is a latency property; any observable difference in
text is a defect.

#### Scenario: A conversation grows by one turn

- **WHEN** a request's prompt extends the previously cached prompt
- **THEN** only the new tokens are decoded
- **AND** the response is identical to the response the same request would produce with an
  empty cache

#### Scenario: A caller requires an independent call

- **WHEN** a request disables cache reuse
- **THEN** the cache is cleared before the prompt is decoded
- **AND** behaviour is identical to that of a freshly created engine

#### Scenario: A new prompt shares no prefix with the cached one

- **WHEN** the longest common prefix between the cached sequence and the new prompt is empty
- **THEN** the whole prompt is decoded
- **AND** cost is no worse than clearing the cache on every call

#### Scenario: Reuse is measured

- **WHEN** prefill cost is measured across a growing conversation
- **THEN** the measurement synchronizes with the backend before reading the clock, because
  decoding is asynchronous on Metal and CUDA and an unsynchronized measurement reports only
  enqueue latency
