// Package natives embeds the modelnexus native closure so that loading the
// library needs no network access.
//
// Import it for effect and nothing else changes:
//
//	import _ "github.com/muthuishere/modelnexus/natives"
//
// It is a SEPARATE MODULE on purpose. Go has no optional dependencies and
// go:embed selects at compile time, not fetch time, so five platforms inside
// bindings/go would put ~70 MB in the module cache of every consumer — including
// the ones who only ever set MODELNEXUS_LIB. A build tag would not help either:
// a tagged file's imports still land in go.mod, which is the same reason the S3
// model resolver lives in its own module. See ADR-0010.
//
// The binary carries only the running platform: the embed directives are behind
// build tags, so the module is large and the executable is not.
//
// "Bundled" means no network. It does not mean no filesystem — a shared library
// cannot be portably loaded from memory, so the closure is written to a cache
// directory before it is opened.
package natives

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	modelnexus "github.com/muthuishere/modelnexus/bindings/go"
)

func init() { modelnexus.RegisterBundle(Dir) }

var (
	once   sync.Once
	dir    string
	dirErr error
)

// Dir materialises the embedded closure and returns the directory holding it.
// The first call extracts; later calls are free. Safe for concurrent use, and
// safe against another process doing the same thing at the same moment.
func Dir() (string, error) {
	once.Do(func() { dir, dirErr = extract() })
	return dir, dirErr
}

// manifest is the links.json every build stages beside the closure. go:embed
// takes regular files only — it does not record symbolic links and does not
// report dropping them — and the bridge links @rpath/libllama.0.dylib, which IS
// one. Without replaying these, the extracted directory looks complete, weighs
// about right, and cannot load (spike 0009).
type manifest struct {
	Links map[string]string `json:"links"`
}

func extract() (string, error) {
	root := "payload/" + platformKey()
	if _, err := fs.Stat(payload, root); err != nil {
		return "", fmt.Errorf("this build of the natives module carries no closure for %s", platformKey())
	}

	// Keyed on the CONTENT, not on a version string. Keying the downloaded natives
	// on the llama.cpp tag alone already served a stale library once, which is why
	// fetch.go had to grow BridgeVersion; a content hash cannot have that bug.
	key, err := contentKey(root)
	if err != nil {
		return "", err
	}
	base, err := cacheRoot()
	if err != nil {
		return "", err
	}
	target := filepath.Join(base, "bundled", key)

	// The stamp is what makes every later process start free. Hashing 14 MB on each
	// start would cost more than the load it protects.
	stamp := filepath.Join(target, ".complete")
	if _, err := os.Stat(stamp); err == nil {
		return target, nil
	}

	tmp := fmt.Sprintf("%s.tmp-%d", target, os.Getpid())
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return "", fmt.Errorf("cannot write the bundled closure to %s: %w", filepath.Dir(target), err)
	}
	defer os.RemoveAll(tmp)

	var man manifest
	if b, err := fs.ReadFile(payload, root+"/links.json"); err == nil {
		if err := json.Unmarshal(b, &man); err != nil {
			return "", fmt.Errorf("the embedded link manifest is unreadable: %w", err)
		}
	}

	err = fs.WalkDir(payload, root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := fs.ReadFile(payload, p)
		if err != nil {
			return err
		}
		// 0o755: these are loadable objects and some loaders care. Cheap to get
		// right, miserable to diagnose.
		return os.WriteFile(filepath.Join(tmp, filepath.Base(p)), b, 0o755)
	})
	if err != nil {
		return "", err
	}

	// Replay the links. Order is irrelevant: a link may legally point at a name
	// that does not exist yet, and llama.cpp chains them
	// (libggml.dylib -> libggml.0.dylib -> libggml.0.13.0.dylib).
	for name, tgt := range man.Links {
		p := filepath.Join(tmp, name)
		_ = os.Remove(p)
		if err := os.Symlink(tgt, p); err != nil {
			return "", fmt.Errorf("could not recreate %s -> %s: %w", name, tgt, err)
		}
	}

	if err := os.WriteFile(stampPath(tmp), []byte(key), 0o644); err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, target); err != nil {
		// Another process extracted the identical bytes first. That is success --
		// the content key guarantees it wrote the same closure.
		if _, serr := os.Stat(stamp); serr == nil {
			return target, nil
		}
		return "", err
	}
	return target, nil
}

func stampPath(dir string) string { return filepath.Join(dir, ".complete") }

func contentKey(root string) (string, error) {
	h := sha256.New()
	err := fs.WalkDir(payload, root, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		b, err := fs.ReadFile(payload, p)
		if err != nil {
			return err
		}
		h.Write([]byte(p))
		h.Write(b)
		return nil
	})
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil))[:16], nil
}

// cacheRoot honours MODELNEXUS_CACHE, as the downloaded natives already do, so a
// machine with one cache policy has one cache policy.
func cacheRoot() (string, error) {
	if env := os.Getenv("MODELNEXUS_CACHE"); env != "" {
		return env, nil
	}
	d, err := os.UserCacheDir()
	if err != nil {
		return "", errors.New("no user cache directory; set MODELNEXUS_CACHE to a writable path")
	}
	return filepath.Join(d, "modelnexus"), nil
}

func platformKey() string {
	arch := runtime.GOARCH
	switch arch {
	case "amd64":
		arch = "x86_64"
	case "arm64":
		arch = "aarch64"
	}
	return runtime.GOOS + "-" + arch
}
