// loadprobe answers one question about a native closure: does it load and
// generate on this machine?
//
// That is a deliberately low bar, and it is the bar this project keeps failing.
// `lipo -archs` and a green compile both say the library is the right shape;
// neither says it works. Three Windows natives shipped green on exactly that
// evidence, and the Windows ARM one loaded cleanly before dying inside ggml's
// compute kernels with 0xC0000409 — so loading is not enough either. Only
// producing a token exercises the path that broke.
//
// It exists as a real command rather than a test because it must run against a
// STAGED closure on a machine that may have nothing else: a CI runner checking a
// cross-build, a bare Windows box, someone else's laptop.
//
//	go build -o loadprobe ./cmd/loadprobe
//	MODELNEXUS_LIB=../../core/dist/darwin-x86_64 ./loadprobe model.gguf
//
// Exits non-zero on any failure, so it composes in a workflow without parsing.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func main() {
	wantArch := flag.String("arch", "", "fail unless GOARCH is this (e.g. amd64), for cross-build checks")
	flag.Parse()

	fmt.Printf("loadprobe on %s/%s\n", runtime.GOOS, runtime.GOARCH)
	if *wantArch != "" && runtime.GOARCH != *wantArch {
		die("this binary is %s, expected %s — it would not exercise the target architecture",
			runtime.GOARCH, *wantArch)
	}

	version, err := modelnexus.Version()
	if err != nil {
		// On Windows this is where a missing sibling DLL surfaces; on Unix it is
		// where a dropped symlink does, because the bridge resolves
		// @rpath/libllama.0.dylib and that name IS a symlink (ADR-0010).
		die("the native library did not load: %v", err)
	}
	fmt.Println("ok    loads    ", version)

	if flag.NArg() < 1 {
		fmt.Println("\nno model given — the library loads, but nothing has generated a token here.")
		fmt.Println("that is half the check. pass a .gguf path to run the other half.")
		os.Exit(1)
	}
	model := flag.Arg(0)
	if _, err := os.Stat(model); err != nil {
		die("model not found: %s", model)
	}

	_ = modelnexus.SetLogLevel(modelnexus.LogError)
	chat, err := modelnexus.Open(model, modelnexus.WithContextSize(1024))
	if err != nil {
		die("Open: %v", err)
	}
	defer chat.Close()

	n, seed, temp := 16, uint32(42), 0.0
	res, err := chat.Infer(modelnexus.Request{
		Messages:    []modelnexus.Message{{Role: "user", Content: "Name the capital of France in one word."}},
		MaxTokens:   &n,
		Seed:        &seed,
		Temperature: &temp,
	})
	if err != nil {
		die("Infer: %v", err)
	}
	// Any token would prove the kernels ran; a CORRECT one also proves the closure
	// is not a mismatched set of libraries that happen to link.
	if !strings.Contains(res.Text, "Paris") {
		die("wrong answer: %q", res.Text)
	}
	fmt.Println("ok    generates", strings.TrimSpace(res.Text))
}

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FAIL  "+format+"\n", args...)
	os.Exit(1)
}
