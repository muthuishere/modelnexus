module citenexus-example

go 1.26

require (
	github.com/muthuishere/citenexus/golang v0.0.0
	github.com/muthuishere/modelnexus/bindings/go v0.2.1
)

require github.com/ebitengine/purego v0.9.0 // indirect

replace github.com/muthuishere/modelnexus/bindings/go => ../../../bindings/go

replace github.com/muthuishere/citenexus/golang => ../../../../rag-cite-nexus/golang
