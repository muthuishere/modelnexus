package modelnexus

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ErrNativeLibraryNotFound reports that the native bridge could not be located or
// loaded.
//
// Go is the one binding that does not receive the shared library from a package
// manager -- purego resolves it at runtime -- so this failure mode is unique to Go
// and is surfaced as a distinct typed error rather than a panic. See ADR-0002 and
// the model-core spec.
type ErrNativeLibraryNotFound struct {
	Searched []string
	Cause    error
}

func (e *ErrNativeLibraryNotFound) Error() string {
	var b strings.Builder
	b.WriteString("modelnexus: native bridge not found or not loadable\n")
	if e.Cause != nil {
		b.WriteString("  cause: " + e.Cause.Error() + "\n")
	}
	b.WriteString("  looked in:\n")
	for _, p := range e.Searched {
		b.WriteString("    " + p + "\n")
	}
	b.WriteString("  build it with core/build.sh, or set MODELNEXUS_LIB to the library or its directory")
	return b.String()
}

func (e *ErrNativeLibraryNotFound) Unwrap() error { return e.Cause }

// PlatformKey is the os-arch key used for the staged native directory layout.
func PlatformKey() string {
	arch := runtime.GOARCH
	switch arch {
	case "amd64":
		arch = "x86_64"
	case "arm64":
		arch = "aarch64"
	}
	return runtime.GOOS + "-" + arch
}

func libFilename() string {
	switch runtime.GOOS {
	case "darwin":
		return "libllamabridge.dylib"
	case "windows":
		return "llamabridge.dll"
	default:
		return "libllamabridge.so"
	}
}

// bindAll resolves every symbol this binding needs, converting purego's panic on a
// missing symbol into an error. A missing symbol means the library on disk predates
// this binding — the useful thing to say, rather than "dlsym: symbol not found".
func bindAll(handle uintptr) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf(
				"native library is missing a symbol this binding needs (%v) — it is "+
					"older than modelnexus %s; delete the cache and call Fetch() again, "+
					"or point MODELNEXUS_LIB at a current build", r, BridgeVersion)
		}
	}()
	purego.RegisterLibFunc(&llbVersion, handle, "llb_version")
	purego.RegisterLibFunc(&llbModelInfo, handle, "llb_model_info")
	purego.RegisterLibFunc(&llbChatCreate, handle, "llb_chat_create")
	purego.RegisterLibFunc(&llbChatInfer, handle, "llb_chat_infer")
	purego.RegisterLibFunc(&llbChatStream, handle, "llb_chat_infer_stream")
	purego.RegisterLibFunc(&llbCountTokens, handle, "llb_count_tokens")
	purego.RegisterLibFunc(&llbChatCache, handle, "llb_chat_cache")
	purego.RegisterLibFunc(&llbStringFree, handle, "llb_string_free")
	purego.RegisterLibFunc(&llbChatDestroy, handle, "llb_chat_destroy")
	purego.RegisterLibFunc(&llbChatLora, handle, "llb_chat_lora")
	purego.RegisterLibFunc(&llbEmbedCreate, handle, "llb_embed_create")
	purego.RegisterLibFunc(&llbEmbed, handle, "llb_embed")
	purego.RegisterLibFunc(&llbRerank, handle, "llb_rerank")
	purego.RegisterLibFunc(&llbEmbedDestroy, handle, "llb_embed_destroy")
	purego.RegisterLibFunc(&llbSetLogLevel, handle, "llb_set_log_level")
	purego.RegisterLibFunc(&llbSetLogCallback, handle, "llb_set_log_callback")
	return nil
}

// candidates lists where to look, most explicit first: an explicit override, then
// the repo's own build output for developers working in the tree, then whatever
// Fetch has already downloaded.
//
// The cache entry is what makes Fetch a whole feature rather than half of one.
// Without it, Fetch downloads the library, returns the directory, and the very
// next Open still fails -- the caller has to hand the path back through
// MODELNEXUS_LIB, which no documentation could make feel like anything but a bug.
func candidates() []string {
	name := libFilename()
	var out []string

	if env := os.Getenv("MODELNEXUS_LIB"); env != "" {
		if st, err := os.Stat(env); err == nil && !st.IsDir() {
			out = append(out, env)
		} else {
			out = append(out, filepath.Join(env, name))
		}
	}

	if wd, err := os.Getwd(); err == nil {
		dir := wd
		// Walk up looking for core/dist/<platform>/<lib>; the binding may be invoked
		// from anywhere inside the repo.
		for i := 0; i < 6; i++ {
			out = append(out, filepath.Join(dir, "core", "dist", PlatformKey(), name))
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	// Fetch's cache, last: an explicit override and a local build both win, so a
	// developer working in the tree is never silently served a downloaded library
	// instead of the one they just compiled.
	if dir, err := CacheDir(); err == nil {
		out = append(out, filepath.Join(dir, PlatformKey(), name))
	}

	return out
}

// The ABI, bound once. Signatures mirror core/include/llamabridge.h exactly.
//
// Every function returning a string the caller owns is declared as unsafe.Pointer,
// never as string: purego would copy a string return and discard the pointer, making
// llb_string_free impossible to call and leaking every single result.
//
// Opaque handles stay uintptr on purpose -- they are never dereferenced here, only
// handed back to the core, so there is nothing for the collector to track.
var (
	loadOnce sync.Once
	loadErr  error

	llbVersion     func() unsafe.Pointer
	llbModelInfo   func(path string) unsafe.Pointer
	llbChatCreate  func(path string, config string, eventCB uintptr, userData uintptr) uintptr
	llbChatInfer   func(chat uintptr, req string) unsafe.Pointer
	llbChatStream  func(chat uintptr, req string, tokenCB uintptr, userData uintptr) unsafe.Pointer
	llbCountTokens func(chat uintptr, req string) unsafe.Pointer
	llbChatCache   func(chat uintptr, req string) unsafe.Pointer
	llbStringFree  func(s unsafe.Pointer)
	llbChatDestroy func(chat uintptr)

	llbChatLora     func(chat uintptr, req string) unsafe.Pointer
	llbEmbedCreate  func(path string, config string, eventCB uintptr, userData uintptr) uintptr
	llbEmbed        func(embed uintptr, req string) unsafe.Pointer
	llbRerank       func(embed uintptr, req string) unsafe.Pointer
	llbEmbedDestroy func(embed uintptr)

	llbSetLogLevel    func(level int32)
	llbSetLogCallback func(cb uintptr, userData uintptr)
)

func ensureLoaded() error {
	loadOnce.Do(func() {
		searched := candidates()
		var lastErr error
		for _, path := range searched {
			if st, err := os.Stat(path); err != nil || st.IsDir() {
				continue
			}
			handle, err := loadLibrary(path)
			if err != nil {
				// Present but unloadable is a different problem from absent, and far
				// more confusing to debug. Keep it and report it.
				lastErr = fmt.Errorf("%s: %w", path, err)
				continue
			}
			// purego PANICS on a missing symbol. A native library that is present
			// and loadable but OLDER than this binding is the likely cause, and a
			// raw dlsym panic is a terrible way to learn that. Turn it into the
			// typed error the caller can actually act on.
			if err := bindAll(handle); err != nil {
				lastErr = fmt.Errorf("%s: %w", path, err)
				continue
			}
			return
		}
		loadErr = &ErrNativeLibraryNotFound{Searched: searched, Cause: lastErr}
	})
	return loadErr
}

// goString copies a NUL-terminated C string into Go without taking ownership.
//
// The ABI above is declared in terms of unsafe.Pointer rather than uintptr for
// exactly this reason: a uintptr round-trip is what `go vet` flags as "possible
// misuse of unsafe.Pointer", and it is a real hazard, not a false positive -- a
// uintptr is an integer the garbage collector will not keep alive. Keeping every
// native address as unsafe.Pointer end to end makes the conversion legal and the
// vet check meaningful instead of suppressed.
func goString(ptr unsafe.Pointer) string {
	if ptr == nil {
		return ""
	}
	var n int
	for *(*byte)(unsafe.Add(ptr, n)) != 0 {
		n++
	}
	if n == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(ptr), n))
}

// takeString copies a core-owned string and frees the original.
//
// Doing the copy and the free together is what stops the free being forgotten at
// one of the several call sites that need it.
func takeString(ptr unsafe.Pointer) string {
	if ptr == nil {
		return ""
	}
	s := goString(ptr)
	llbStringFree(ptr)
	return s
}
