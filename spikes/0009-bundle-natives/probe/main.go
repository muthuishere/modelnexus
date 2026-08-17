// SPIKE 0009 / question 1, second half: the cross-built darwin-x86_64 native
// COMPILES -- but does it LOAD AND GENERATE?
//
// That distinction is the entire lesson of the Windows work: natives.yml went
// green three times on a library that could not load. `lipo -archs` proving
// x86_64 is the same weak claim. So this binary is built GOARCH=amd64 and run
// under Rosetta 2, which makes an Apple Silicon Mac an honest Intel test bed.
package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func main() {
	fmt.Printf("probe running as %s/%s\n", runtime.GOOS, runtime.GOARCH)
	if runtime.GOARCH == "" {
		fmt.Println("FAIL: not an amd64 binary; the whole point is to exercise x86_64")
		os.Exit(1)
	}

	v, err := modelnexus.Version()
	if err != nil {
		fmt.Println("FAIL: the x86_64 native did not load:", err)
		os.Exit(1)
	}
	fmt.Println("ok    loads   ", v)

	model := os.Args[1]
	_ = modelnexus.SetLogLevel(modelnexus.LogError)
	chat, err := modelnexus.Open(model, modelnexus.WithContextSize(1024))
	if err != nil {
		fmt.Println("FAIL: Open:", err)
		os.Exit(1)
	}
	defer chat.Close()

	// Generation is what exercises the ggml compute kernels -- the exact place the
	// Windows ARM native died with 0xC0000409 AFTER loading cleanly.
	n, seed, temp := 16, uint32(42), 0.0
	res, err := chat.Infer(modelnexus.Request{
		Messages:    []modelnexus.Message{{Role: "user", Content: "Name the capital of France in one word."}},
		MaxTokens:   &n,
		Seed:        &seed,
		Temperature: &temp,
	})
	if err != nil {
		fmt.Println("FAIL: Infer:", err)
		os.Exit(1)
	}
	if !strings.Contains(res.Text, "Paris") {
		fmt.Printf("FAIL: wrong answer: %q\n", res.Text)
		os.Exit(1)
	}
	fmt.Println("ok    generates", strings.TrimSpace(res.Text))
	fmt.Println("\nVERDICT: a cross-built darwin-x86_64 native loads and runs.")
}
