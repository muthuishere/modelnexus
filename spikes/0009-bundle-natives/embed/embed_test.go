// SPIKE 0009 / question 2: can Go embed the native closure, extract it, and
// dlopen the result?
//
// The hazard is NOT size. It is SYMLINKS. dist/ has 18 of them, because
// llama.cpp ships libfoo.dylib -> libfoo.0.dylib -> libfoo.0.0.N.dylib and
// build.sh preserves them with `cp -a` deliberately -- dereferencing turns one
// 7.5 MB library into three (core/build.sh:125).
//
// go:embed walks with fs.WalkDir and takes REGULAR FILES ONLY. It does not
// record symlinks and does not error about them; it silently omits them. So the
// question is whether what survives embedding is still loadable, or whether the
// extracted directory is missing exactly the names the bridge resolves at
// runtime via @rpath.
package main

import (
	"embed"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

//go:embed native
var native embed.FS

func TestEmbedPreservesWhatIsNeeded(t *testing.T) {
	// What is on disk.
	var onDiskFiles, onDiskLinks []string
	root := "native/darwin-aarch64"
	err := filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			onDiskLinks = append(onDiskLinks, filepath.Base(p))
		} else {
			onDiskFiles = append(onDiskFiles, filepath.Base(p))
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// filepath.Walk uses Lstat, so symlinks land in onDiskLinks.

	// What survived the embed.
	var embedded []string
	_ = fs.WalkDir(native, ".", func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			embedded = append(embedded, filepath.Base(p))
		}
		return nil
	})

	fmt.Printf("on disk : %d regular files, %d symlinks\n", len(onDiskFiles), len(onDiskLinks))
	fmt.Printf("embedded: %d entries\n", len(embedded))

	have := map[string]bool{}
	for _, e := range embedded {
		have[e] = true
	}
	missing := []string{}
	for _, l := range onDiskLinks {
		if !have[l] {
			missing = append(missing, l)
		}
	}
	fmt.Printf("symlink names LOST by go:embed: %d\n", len(missing))
	for _, m := range missing {
		fmt.Println("   ", m)
	}
}
