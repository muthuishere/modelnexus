module github.com/muthuishere/modelnexus/examples/go

go 1.23

require github.com/muthuishere/modelnexus/bindings/go v0.0.0

require github.com/ebitengine/purego v0.9.0 // indirect

// These examples are run from inside the repo, against the binding in this tree, so
// that a change to the binding breaks them in the same commit. A consumer outside the
// repo drops this line and gets the published module:
//
//	go get github.com/muthuishere/modelnexus/bindings/go
replace github.com/muthuishere/modelnexus/bindings/go => ../../bindings/go
