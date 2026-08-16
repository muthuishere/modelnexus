# model-core

## ADDED Requirements

### Requirement: A binding rejects what it does not understand

A binding SHALL NOT silently forward a caller parameter it does not recognise. An unrecognised
named parameter SHALL be reported to the caller as an error in that language's idiom.

A binding MAY offer an explicit escape hatch for reaching a core parameter the binding has not
named, provided the escape hatch cannot be invoked by a typo.

#### Scenario: A caller misspells a parameter

- **WHEN** a caller passes a named parameter the binding does not recognise
- **THEN** the call fails with an error naming the unrecognised parameter
- **AND** no request is sent to the core

#### Scenario: A caller deliberately reaches an unnamed core parameter

- **WHEN** a caller uses the binding's explicit pass-through for an unnamed core parameter
- **THEN** the parameter is sent unchanged
- **AND** no binding change was required to reach it

### Requirement: A binding has one naming convention

A binding SHALL present a single naming convention across its whole surface — requests and
responses alike — and SHALL translate at the boundary between that convention and the wire
format. No field SHALL be exempt.

Caller-supplied JSON Schemas SHALL be passed through untranslated, because their property names
belong to the caller and describe the output contract rather than the binding's own surface.

#### Scenario: A caller builds a request

- **WHEN** a caller sets any generation parameter
- **THEN** every parameter uses the same naming convention as every other
- **AND** the binding translates them to the wire format

#### Scenario: A caller reads a response

- **WHEN** a caller reads any response field
- **THEN** it uses the same naming convention as the request surface

#### Scenario: A caller supplies a JSON Schema

- **WHEN** a request carries a caller-supplied JSON Schema
- **THEN** its property names reach the core exactly as written
- **AND** the binding does not rename, reorder, or otherwise rewrite them

### Requirement: Defaults live in the core, and nowhere else

A binding SHALL send only configuration the caller explicitly supplied. It SHALL NOT restate a
default the core already owns.

#### Scenario: A caller supplies no configuration

- **WHEN** an engine is created with no configuration
- **THEN** the binding sends no configuration at all
- **AND** the core's own defaults apply

#### Scenario: A caller supplies one setting

- **WHEN** an engine is created with a single configuration value
- **THEN** only that value is sent
- **AND** every other setting is left to the core

#### Scenario: The core changes a default

- **WHEN** the core changes a default value
- **THEN** every binding follows it without a binding change

### Requirement: The parity gate covers creation, not only inference

The cross-binding conformance gate SHALL compare what each binding sends at engine-creation
time, not only what each binding returns from inference. Bindings SHALL produce identical
create-configuration for identical caller intent.

The gate SHALL also assert that supplied configuration is observable through the ABI, so that a
binding which marshals configuration and discards it fails rather than appearing correct.

#### Scenario: Two bindings are given the same creation intent

- **WHEN** each binding creates an engine with the same caller intent
- **THEN** the configuration each sends to the core is identical

#### Scenario: A binding drops a configuration value

- **WHEN** a binding accepts a configuration value and does not pass it to the core
- **THEN** the gate fails, because the value is asserted through an observable effect rather
  than assumed from a successful load
