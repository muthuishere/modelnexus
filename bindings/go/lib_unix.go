//go:build !windows

package modelnexus

import "github.com/ebitengine/purego"

// loadLibrary opens the native bridge.
//
// Split per-OS because purego.Dlopen is Unix-only: it lives behind
// `//go:build darwin || freebsd || linux || netbsd` in purego, so a
// `GOOS=windows go build` of this package failed to compile at all. Only
// RegisterLibFunc is portable (`darwin || freebsd || linux || netbsd ||
// windows`), which is why the rest of the binding needed no change.
func loadLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
