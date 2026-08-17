//go:build darwin && amd64

package natives

import "embed"

// Only this platform's closure is embedded in the binary, even though the module
// carries every platform. That is the whole reason the embeds are behind build
// tags: a large module, a normal-sized executable.
//
//go:embed payload/darwin-x86_64
var payload embed.FS
