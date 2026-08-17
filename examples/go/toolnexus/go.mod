module toolnexus-example

go 1.23.0

require (
	github.com/muthuishere/modelnexus/bindings/go v0.2.1
	github.com/muthuishere/toolnexus/golang v0.16.0
)

require (
	github.com/ebitengine/purego v0.9.0 // indirect
	github.com/google/jsonschema-go v0.4.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/mark3labs/mcp-go v0.48.0 // indirect
	github.com/spf13/cast v1.7.1 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/muthuishere/modelnexus/bindings/go => ../../../bindings/go

// NO replace for toolnexus: this example exists to prove the PUBLISHED module works
// against modelnexus. Pointing at a local checkout would prove only that the working
// tree does.
