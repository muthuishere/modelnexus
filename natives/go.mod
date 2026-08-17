module github.com/muthuishere/modelnexus/natives

go 1.23

require github.com/muthuishere/modelnexus/bindings/go v0.2.1

require github.com/ebitengine/purego v0.9.0 // indirect

// Local development: the published module resolves the tagged binding normally.
replace github.com/muthuishere/modelnexus/bindings/go => ../bindings/go
