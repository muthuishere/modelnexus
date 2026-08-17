# model-core

## ADDED Requirements

### Requirement: Every binding resolves the native library through the same ordered steps

A binding SHALL locate its native library by trying, in this order and stopping at the first
success:

1. the path named by `MODELNEXUS_LIB`
2. a closure bundled with the binding's distribution
3. the repository's `core/dist/<platform-key>/`
4. a download from the release keyed by the pinned llama.cpp tag

A binding MAY have no bundled closure available for the running platform, in which case step 2
is a miss rather than an error. A binding SHALL NOT reorder these steps, add a step, or skip a
step it can perform.

Step 4 SHALL NOT be reached when step 2 succeeded, so that a bundled binding performs no
network access to load its library.

#### Scenario: A bundled binding loads on a machine with no network

- **WHEN** a caller loads a binding whose distribution includes a closure for the running platform
- **AND** no network is reachable
- **THEN** the library loads from the bundled closure
- **AND** no network request is attempted

#### Scenario: An explicit path overrides a bundled closure

- **WHEN** `MODELNEXUS_LIB` names a directory containing a loadable library
- **AND** the binding also ships a bundled closure for the running platform
- **THEN** the library named by `MODELNEXUS_LIB` is loaded
- **AND** the bundled closure is not extracted

#### Scenario: A platform has no bundled closure

- **WHEN** a caller loads a binding on a platform its distribution does not bundle
- **THEN** resolution continues to the remaining steps
- **AND** the library is downloaded and loaded if the release provides that platform

#### Scenario: Every step fails

- **WHEN** no step can produce a loadable library
- **THEN** a typed error naming every path and source that was tried is raised in that language's idiom
- **AND** the process does not crash

### Requirement: A materialised closure contains every name the staged closure had

The invariant is **name completeness**, not link preservation. A closure that reaches a
machine SHALL present every file name the build staged, because the bridge resolves siblings
by name at load time and a missing name is an unloadable closure.

A distribution mechanism MAY satisfy this with a symbolic link or with a duplicate file,
whichever its ecosystem can carry safely. Where links can be carried, they SHALL be preferred,
because duplication costs roughly three times the closure size on disk.

Every build SHALL emit a manifest of the symbolic links it staged, in the same shape on every
platform, including platforms with none. A mechanism that cannot carry links SHALL recreate
them from that manifest when it materialises the closure.

A build SHALL NOT stage a closure containing a link whose target is absent.

#### Scenario: A mechanism cannot carry symbolic links

- **WHEN** a bundled closure is materialised by a mechanism that does not preserve links
- **THEN** every name recorded in the manifest exists in the materialised closure
- **AND** the closure loads

#### Scenario: A mechanism can carry symbolic links

- **WHEN** a distribution mechanism can carry symbolic links safely
- **THEN** links are carried rather than duplicated
- **AND** the materialised closure is not larger than the staged one

#### Scenario: The manifest names an entry that cannot be created

- **WHEN** materialisation cannot create an entry the manifest records
- **THEN** it fails with an error naming the entry
- **AND** no partially materialised closure is left where a loader could find it

#### Scenario: A staged closure has a dangling link

- **WHEN** a build stages a closure containing a symbolic link whose target is not present
- **THEN** the build fails and names the link and its target
- **AND** the closure is not published

### Requirement: Materialising a bundled closure is atomic and idempotent

A binding that extracts a bundled closure to disk SHALL write it to a temporary location and
move it into its final location in one step, so that a partially written closure is never
visible at the path a loader reads.

A second materialisation of the same closure SHALL be a no-op. The destination SHALL be keyed
on the content of the closure, so that a binding upgrade cannot load a closure staged by an
earlier one.

Concurrent materialisation of the same closure by more than one process SHALL succeed in every
process.

#### Scenario: Materialisation is interrupted

- **WHEN** a process is killed while extracting a bundled closure
- **AND** another process then loads the same binding
- **THEN** the interrupted output is not loaded
- **AND** the closure is materialised again and loads

#### Scenario: The binding is upgraded

- **WHEN** a binding shipping a different closure runs on a machine holding an earlier one
- **THEN** the earlier closure is not loaded
- **AND** the closure shipped with the running binding is used

#### Scenario: Two processes start at once

- **WHEN** two processes materialise the same closure concurrently
- **THEN** both load the library successfully
- **AND** neither reports a conflict

### Requirement: The supported platform set is built, not seeded by hand

Every platform the project distributes SHALL be produced by the automated native build. A
platform SHALL NOT depend on a manual step by a maintainer to appear in a release.

A release SHALL fail rather than publish a package for a platform whose closure is absent.

#### Scenario: A platform's closure is missing at release time

- **WHEN** a release runs and a supported platform has no staged closure
- **THEN** the release fails and names the missing platform
- **AND** no package is published for any platform

#### Scenario: A platform has no runner of its own architecture

- **WHEN** the supported set includes a platform with no available CI runner of that architecture
- **THEN** its closure is cross-built by an available runner
- **AND** the result is verified by loading it and generating a token, not by inspecting its architecture alone

## MODIFIED Requirements

### Requirement: Go binding reports a missing native runtime explicitly

The Go binding MAY receive its native library from a package (an opt-in bundle module) or load
it at runtime. It SHALL surface an unavailable or unloadable native library as a distinct,
typed error at first use, rather than a panic or a generic failure.

The error SHALL name every location that was tried, including which of the ordered resolution
steps were attempted and why each did not produce a library.

Bundling SHALL be opt-in and SHALL NOT be a dependency of the base binding module, so that a
consumer who does not want it does not download it.

#### Scenario: The shared library is absent

- **WHEN** a Go caller invokes the binding and the native library cannot be located or loaded
- **THEN** a distinct typed error identifying the missing runtime is returned
- **AND** the process does not panic

#### Scenario: A caller opts into the bundle

- **WHEN** a Go caller imports the natives module
- **THEN** the library is loaded from the bundled closure with no network access
- **AND** no change to the caller's other code is required

#### Scenario: A caller does not opt into the bundle

- **WHEN** a Go caller depends only on the base binding module
- **THEN** no bundled closure is downloaded as part of resolving that dependency
- **AND** the binding resolves its library exactly as it did before the bundle existed
