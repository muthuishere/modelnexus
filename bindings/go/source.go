package modelnexus

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Resolve turns a model reference into a local file path, downloading it once if
// it is remote (ADR-0009).
//
//	qwen2.5-1.5b-instruct-q4_k_m.gguf          a path — returned unchanged
//	hf:Qwen/Qwen2.5-1.5B-Instruct-GGUF/x.gguf  Hugging Face
//	https://internal.example/models/x.gguf     anything HTTP(S)
//	s3://bucket/models/x.gguf                  S3 (needs the aws build tag)
//
// Anything that is not a recognised scheme is a path. That ordering is deliberate:
// a local file must never be mistaken for a URI, because the failure would be a
// network call for a file the caller can see.
//
// CREDENTIALS ARE NEVER ARGUMENTS. Hugging Face reads HF_TOKEN from the
// environment; S3 uses the AWS default credential chain. A signature that accepts
// a secret is a signature that ends up in source control.
func Resolve(ctx context.Context, ref string, onProgress func(done, total int64)) (string, error) {
	switch {
	case strings.HasPrefix(ref, "hf:"):
		u, err := huggingFaceURL(strings.TrimPrefix(ref, "hf:"))
		if err != nil {
			return "", err
		}
		return download(ctx, "hf", ref, u, hfHeaders(), onProgress)
	case strings.HasPrefix(ref, "https://"), strings.HasPrefix(ref, "http://"):
		return download(ctx, "http", ref, ref, nil, onProgress)
	case strings.HasPrefix(ref, "s3://"):
		return resolveS3(ctx, ref, onProgress)
	default:
		if _, err := os.Stat(ref); err != nil {
			return "", &Error{Code: "MODEL_NOT_FOUND", Message: "no file at " + ref +
				" (a reference with no scheme is treated as a local path; use hf: / s3:// / https:// for a remote one)"}
		}
		return ref, nil
	}
}

// huggingFaceURL maps owner/repo/path... onto the resolve endpoint.
func huggingFaceURL(spec string) (string, error) {
	parts := strings.SplitN(spec, "/", 3)
	if len(parts) < 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return "", &Error{Code: "INVALID_MODEL_REF", Message: "hf: needs owner/repo/file.gguf, got " + spec}
	}
	owner, repo, file := parts[0], parts[1], parts[2]
	// "main" is the revision. A pinned revision belongs in the ref itself, and
	// that is a decision ADR-0009 leaves open rather than guessing at.
	return fmt.Sprintf("https://huggingface.co/%s/%s/resolve/main/%s?download=true",
		url.PathEscape(owner), url.PathEscape(repo), file), nil
}

func hfHeaders() map[string]string {
	if t := os.Getenv("HF_TOKEN"); t != "" {
		return map[string]string{"Authorization": "Bearer " + t}
	}
	return nil
}

// modelCacheDir is where a resolved model lives. Keyed on a hash of the ORIGINAL
// reference, so two bindings on one machine share the download instead of each
// keeping its own copy of a 4 GB file.
func modelCacheDir(scheme, ref string) (string, error) {
	base, err := CacheDir()
	if err != nil {
		return "", err
	}
	// Deliberately NOT under the llama-tag/bridge-version directory that natives
	// use: a model does not change when the engine does, and re-downloading
	// gigabytes on a patch release would be indefensible.
	root := filepath.Join(filepath.Dir(filepath.Dir(base)), "modelnexus", "models")
	sum := sha256.Sum256([]byte(ref))
	return filepath.Join(root, scheme, hex.EncodeToString(sum[:])[:16]), nil
}

func download(ctx context.Context, scheme, ref, u string, headers map[string]string,
	onProgress func(done, total int64)) (string, error) {

	dir, err := modelCacheDir(scheme, ref)
	if err != nil {
		return "", err
	}
	name := path.Base(strings.SplitN(u, "?", 2)[0])
	if name == "" || name == "/" || name == "." {
		name = "model.gguf"
	}
	final := filepath.Join(dir, name)

	// Already there: do not re-download, and do not re-hash gigabytes to prove it.
	if st, err := os.Stat(final); err == nil && st.Size() > 0 {
		return final, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := (&http.Client{Timeout: 0}).Do(req)
	if err != nil {
		return "", &Error{Code: "MODEL_FETCH_FAILED", Message: "could not reach " + u + ": " + err.Error()}
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		hint := "set HF_TOKEN for a gated repository"
		if scheme != "hf" {
			hint = "the source rejected the request"
		}
		return "", &Error{Code: "MODEL_UNAUTHORIZED",
			Message: fmt.Sprintf("%s returned %d — %s", u, resp.StatusCode, hint)}
	}
	if resp.StatusCode >= 300 {
		return "", &Error{Code: "MODEL_FETCH_FAILED",
			Message: fmt.Sprintf("%s returned %d", u, resp.StatusCode)}
	}

	// Download to a temporary name and RENAME into place. A half-downloaded 4 GB
	// file that is openable as a model is the worst outcome available here: it
	// fails deep inside llama.cpp, long after the cause.
	tmp, err := os.CreateTemp(dir, ".part-*")
	if err != nil {
		return "", err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	total := resp.ContentLength
	var done int64
	buf := make([]byte, 1<<20)
	last := time.Now()
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := tmp.Write(buf[:n]); werr != nil {
				tmp.Close()
				return "", werr
			}
			done += int64(n)
			if onProgress != nil && time.Since(last) > 200*time.Millisecond {
				onProgress(done, total)
				last = time.Now()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			tmp.Close()
			return "", &Error{Code: "MODEL_FETCH_FAILED", Message: "download interrupted: " + rerr.Error()}
		}
		if err := ctx.Err(); err != nil {
			tmp.Close()
			return "", err
		}
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	// A truncated transfer that the server did not report is still a broken file.
	if total > 0 && done != total {
		return "", &Error{Code: "MODEL_FETCH_FAILED",
			Message: "expected " + strconv.FormatInt(total, 10) + " bytes, got " + strconv.FormatInt(done, 10)}
	}
	if onProgress != nil {
		onProgress(done, total)
	}
	if err := os.Rename(tmpName, final); err != nil {
		return "", err
	}
	return final, nil
}

// SchemeResolver turns a reference for one scheme into a local file path.
type SchemeResolver func(ctx context.Context, ref string, onProgress func(done, total int64)) (string, error)

var schemes = map[string]SchemeResolver{}

// RegisterScheme adds support for a URI scheme, so a source that needs a heavy
// SDK can live in a SEPARATE module the consumer opts into.
//
// A Go build tag would NOT have been enough: a tagged file's imports still land
// in go.mod, so every consumer would download the AWS SDK to open a local file.
// A separate module is the only thing that actually keeps the base binding
// dependency-free, and this hook is what makes that possible.
//
//	import _ "github.com/muthuishere/modelnexus/bindings/go/s3"   // s3:// now works
func RegisterScheme(scheme string, fn SchemeResolver) { schemes[scheme] = fn }

func resolveS3(ctx context.Context, ref string, onProgress func(done, total int64)) (string, error) {
	if fn, ok := schemes["s3"]; ok {
		return fn(ctx, ref, onProgress)
	}
	return "", &Error{Code: "SCHEME_NOT_REGISTERED", Message: "s3:// needs the S3 source module — " +
		`import _ "github.com/muthuishere/modelnexus/bindings/go/s3"`}
}
