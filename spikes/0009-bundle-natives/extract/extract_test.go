package main

import (
	"embed"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

//go:embed native
var native embed.FS

// The claim under test is end-to-end: embed -> extract -> the OS loader can
// actually resolve the closure. Anything short of dlopen is the same weak
// evidence that let three broken Windows natives ship green.
func TestExtractedClosureLoads(t *testing.T) {
	root := "native/darwin-aarch64"
	dir := filepath.Join(t.TempDir(), "nx", cacheKey(native, root))

	got, err := Extract(native, root, dir)
	if err != nil {
		t.Fatal(err)
	}

	var files, links int
	ents, _ := os.ReadDir(got)
	for _, e := range ents {
		info, _ := e.Info()
		if info.Mode()&os.ModeSymlink != 0 {
			links++
		} else if !e.IsDir() {
			files++
		}
	}
	fmt.Printf("extracted: %d regular files, %d symlinks -> %s\n", files, links, got)
	if links != 18 {
		t.Fatalf("expected 18 symlinks replayed, got %d", links)
	}

	// Second call must be a no-op, not a re-extract.
	again, err := Extract(native, root, dir)
	if err != nil || again != got {
		t.Fatalf("second Extract was not idempotent: %v", err)
	}
	fmt.Println("idempotent: yes")

	// The real proof: hand the directory to the Go binding as MODELNEXUS_LIB and
	// make it generate a token.
	model := os.Getenv("SPIKE_MODEL")
	if model == "" {
		t.Skip("set SPIKE_MODEL to a .gguf to run the load half")
	}
	cmd := exec.Command("./probe.bin", model)
	cmd.Env = append(os.Environ(), "MODELNEXUS_LIB="+got)
	out, err := cmd.CombinedOutput()
	fmt.Println(strings.TrimSpace(string(out)))
	if err != nil {
		t.Fatalf("the extracted closure did not load: %v", err)
	}
}
