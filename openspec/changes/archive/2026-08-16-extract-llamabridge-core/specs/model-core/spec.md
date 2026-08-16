# model-core

## ADDED Requirements

### Requirement: Language-neutral C ABI

The core SHALL expose all inference capability through a pure C ABI using opaque handles, with
no type from any host language runtime crossing the boundary. All request and response detail
SHALL travel as JSON strings, so that adding a generation parameter does not change the ABI.

#### Scenario: A new generation parameter is added

- **WHEN** a new optional generation parameter is introduced
- **THEN** it is accepted inside the request JSON
- **AND** no exported C symbol changes signature
- **AND** no binding requires a code change to pass it through

#### Scenario: A host runtime type is proposed for the boundary

- **WHEN** a change would place a language-specific type in an exported signature
- **THEN** the change is rejected, because the surface must remain bindable from any FFI

### Requirement: Explicit memory ownership

Every pointer the core returns to a caller SHALL have a documented lifetime and an explicit
release function. The core SHALL NOT rely on any caller having a garbage collector.

#### Scenario: An inference result is returned

- **WHEN** `llb_chat_infer` returns a response string
- **THEN** the string is heap-allocated by the core
- **AND** the caller releases it via `llb_string_free`
- **AND** calling the release function with NULL is safe

### Requirement: Inference failures are data, not null

Inference entry points SHALL NOT return NULL to signal failure. Failures SHALL be returned as
error JSON carrying a stable `code` and a human-readable `message`.

#### Scenario: Inference fails

- **WHEN** an inference call fails for any reason
- **THEN** a non-NULL JSON string is returned with `"type": "error"`
- **AND** it carries a stable machine-readable `error.code`

#### Scenario: Two bindings observe the same failure

- **WHEN** the same failure occurs under the Java binding and the Go binding
- **THEN** both report the identical `error.code`
- **AND** each presents it in its own idiom, as an exception and as an error value respectively

### Requirement: Bindings add no behaviour

A language binding SHALL marshal, call, and free only. It SHALL NOT add behaviour, reinterpret
a core result, cache state the core does not hold, or locally work around a core defect.

#### Scenario: A binding needs a capability the ABI lacks

- **WHEN** a language needs behaviour the C ABI does not provide
- **THEN** the capability is added to the ABI, the spec, and every binding
- **AND** it is not implemented inside a single binding

### Requirement: Go binding reports a missing native runtime explicitly

The Go binding loads the shared library at runtime rather than receiving it from a package
manager. It SHALL surface an unavailable or unloadable native library as a distinct, typed
error at first use, rather than a panic or a generic failure.

#### Scenario: The shared library is absent

- **WHEN** a Go caller invokes the binding and the native library cannot be located or loaded
- **THEN** a distinct typed error identifying the missing runtime is returned
- **AND** the process does not panic
