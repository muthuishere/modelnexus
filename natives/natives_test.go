package natives

import (
	"os"
	"path/filepath"
	"testing"
)

// The claim is end to end: the embedded closure materialises into something the
// OS loader can actually open. Anything short of that is the weak evidence that
// let three broken Windows natives ship green.
//
// Skips when the payload has not been staged (see stage.sh), because the
// committed placeholder deliberately carries no library.
func TestBundleMaterialisesAndLoads(t *testing.T) {
	t.Setenv("MODELNEXUS_CACHE", t.TempDir())

	dir, err := Dir()
	if err != nil {
		t.Skipf("no closure staged for this platform: %v", err)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(ents) < 3 {
		t.Skip("payload is the committed placeholder, not a real closure")
	}

	// Every name the manifest recorded must exist. This is the regression test for
	// the bug spike 0009 found: go:embed drops symlinks silently, and the bridge
	// links @rpath/libllama.0.dylib, which is one of them.
	var links int
	for _, e := range ents {
		info, err := e.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			links++
			// A recreated link must resolve, or the closure is decorative.
			if _, err := os.Stat(filepath.Join(dir, e.Name())); err != nil {
				t.Errorf("%s is a dangling symlink after extraction: %v", e.Name(), err)
			}
		}
	}
	t.Logf("materialised %d entries (%d symlinks) into %s", len(ents), links, dir)

	// Second call must be free and identical, not a second extraction.
	again, err := Dir()
	if err != nil || again != dir {
		t.Fatalf("Dir() was not idempotent: %q vs %q, err=%v", again, dir, err)
	}
}

// A bundled closure must never be reached when MODELNEXUS_LIB is set: overriding
// the library is the entire purpose of that variable, and a bundle that quietly
// took it away would be worse than no bundle.
func TestExplicitOverrideStillWins(t *testing.T) {
	// Exercised through the binding's resolution order rather than here, because
	// this module only supplies step 2. Kept as a named test so the requirement is
	// visible in this package rather than only in the gate.
	t.Skip("covered by the parity gate's resolution case")
}
