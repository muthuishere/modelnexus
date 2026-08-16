package modelnexus

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// LlamaTag is the llama.cpp release this binding's native library is built against.
//
// It must match core/build.sh's pin: the natives are published in a GitHub release
// keyed by this tag, and a mismatch means Fetch downloads a library the bindings were
// never tested with.
const LlamaTag = "b9371"

// BridgeVersion is the modelnexus C ABI this binding speaks.
//
// It is part of the cache path, and that is not cosmetic. The natives release is
// keyed on the llama.cpp tag (ADR-0004), but the BRIDGE moves independently: 0.2.0
// added entry points against the same llama.cpp b9371. Keyed on LlamaTag alone, a
// user who fetched 0.1.0 natives would keep a cache that looks valid forever and
// would never re-fetch — a new binding silently loading an old library.
const BridgeVersion = "0.2.0"

// nativesRepo is where the tier-1 workflow parks the per-platform native closure.
const nativesRepo = "muthuishere/modelnexus"

// CacheDir reports where Fetch stores downloaded natives.
//
// Honours MODELNEXUS_CACHE, else the OS user cache directory.
func CacheDir() (string, error) {
	if dir := os.Getenv("MODELNEXUS_CACHE"); dir != "" {
		return dir, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("modelnexus: no user cache directory: %w", err)
	}
	return filepath.Join(base, "modelnexus", LlamaTag+"-"+BridgeVersion), nil
}

// Fetch downloads this platform's native library into the cache, if it is not already
// there, and returns the directory holding it.
//
// Go is the one binding that does not receive the library from its package manager: a
// Go module is a source tree, and embedding five platforms' binaries would make every
// `go get` pull ~70 MB of libraries the user will not use (ADR-0007). So the library
// is fetched once, at the user's explicit request, rather than smuggled into the module.
//
// The returned directory is also searched automatically on the next Open, so the
// usual shape is simply:
//
//	if _, err := modelnexus.Fetch(); err != nil { ... }
//	chat, err := modelnexus.Open("model.gguf")
//
// You do not have to pass the path anywhere.
//
// Fetch is a convenience, not a requirement. Setting MODELNEXUS_LIB to a directory you
// already have — an air-gapped machine, a vendored copy, a build from core/build.sh —
// works and skips this entirely.
func Fetch() (string, error) {
	dir, err := CacheDir()
	if err != nil {
		return "", err
	}
	target := filepath.Join(dir, PlatformKey())

	if _, err := os.Stat(filepath.Join(target, libFilename())); err == nil {
		return target, nil // already cached
	}

	url := fmt.Sprintf("https://github.com/%s/releases/download/natives-%s/natives-%s.zip",
		nativesRepo, LlamaTag, PlatformKey())

	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("modelnexus: could not download natives from %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf(
			"modelnexus: could not download natives from %s: HTTP %d.\n"+
				"If this platform has no published build, build it yourself with core/build.sh "+
				"and point MODELNEXUS_LIB at core/dist/%s",
			url, resp.StatusCode, PlatformKey())
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("modelnexus: could not create %s: %w", dir, err)
	}

	// Download to a temp file first: a zip reader needs random access, and a partial
	// file left at the final path would look like a valid cache on the next run.
	tmp, err := os.CreateTemp(dir, "natives-*.zip")
	if err != nil {
		return "", fmt.Errorf("modelnexus: could not create a temp file: %w", err)
	}
	defer os.Remove(tmp.Name())
	size, err := io.Copy(tmp, resp.Body)
	if err != nil {
		tmp.Close()
		return "", fmt.Errorf("modelnexus: download failed: %w", err)
	}
	tmp.Close()

	if err := unzipInto(tmp.Name(), size, dir); err != nil {
		return "", err
	}
	if _, err := os.Stat(filepath.Join(target, libFilename())); err != nil {
		return "", fmt.Errorf("modelnexus: the downloaded archive did not contain %s", libFilename())
	}
	return target, nil
}

func unzipInto(path string, size int64, dest string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("modelnexus: could not reopen the download: %w", err)
	}
	defer f.Close()

	zr, err := zip.NewReader(f, size)
	if err != nil {
		return fmt.Errorf("modelnexus: the download is not a valid zip: %w", err)
	}

	for _, entry := range zr.File {
		// Reject path traversal rather than trusting an archive to stay inside dest.
		out := filepath.Join(dest, entry.Name) //nolint:gosec // checked immediately below
		if !strings.HasPrefix(filepath.Clean(out), filepath.Clean(dest)+string(os.PathSeparator)) {
			return fmt.Errorf("modelnexus: archive entry escapes the cache directory: %s", entry.Name)
		}

		info := entry.FileInfo()
		if info.IsDir() {
			if err := os.MkdirAll(out, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
			return err
		}

		rc, err := entry.Open()
		if err != nil {
			return err
		}

		// llama.cpp ships versioned aliases as symlinks; recreating them as symlinks
		// keeps the extracted set the same size and shape as the staged one.
		if info.Mode()&os.ModeSymlink != 0 {
			target, err := io.ReadAll(rc)
			rc.Close()
			if err != nil {
				return err
			}
			os.Remove(out)
			if err := os.Symlink(string(target), out); err != nil {
				return err
			}
			continue
		}

		dst, err := os.OpenFile(out, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, info.Mode())
		if err != nil {
			rc.Close()
			return err
		}
		_, err = io.Copy(dst, rc) //nolint:gosec // archive is our own published artifact
		rc.Close()
		dst.Close()
		if err != nil {
			return err
		}
	}
	return nil
}
