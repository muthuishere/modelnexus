//go:build windows

package modelnexus

import (
	"fmt"
	"syscall"
)

// loadLibrary opens the native bridge on Windows.
//
// purego has no Dlopen here — that symbol is Unix-only — but its
// RegisterLibFunc does support Windows, so a raw HMODULE from LoadLibrary is
// all the rest of the binding needs.
//
// LoadLibraryEx with LOAD_WITH_ALTERED_SEARCH_PATH rather than plain
// LoadLibrary: the bridge links llama and ggml as siblings in its own
// directory, and the default search order looks in the PROCESS working
// directory instead. Without this flag the bridge itself loads and then fails
// to resolve its dependencies, which surfaces as a bare "The specified module
// could not be found" naming the file that was found.
func loadLibrary(path string) (uintptr, error) {
	const loadWithAlteredSearchPath = 0x00000008

	p, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	kernel32, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		return 0, err
	}
	proc, err := syscall.GetProcAddress(kernel32, "LoadLibraryExW")
	if err != nil {
		return 0, err
	}
	h, _, errno := syscall.SyscallN(proc, uintptr(unsafePointerOf(p)), 0, loadWithAlteredSearchPath)
	if h == 0 {
		// ERROR_MOD_NOT_FOUND names the file we asked for, not the dependency
		// that is actually missing -- which is why it reads as "the DLL you can
		// see is not there". Measured on a fresh Windows 11: the missing module
		// was the Visual C++ runtime, and nothing in the message said so.
		const errModNotFound syscall.Errno = 126
		if errno == errModNotFound {
			return 0, fmt.Errorf(
				"%s was found but a DEPENDENCY of it could not be loaded. "+
					"On a clean Windows this is almost always the Visual C++ Redistributable "+
					"(VCRUNTIME140.dll / MSVCP140.dll): install "+
					"https://aka.ms/vs/17/release/vc_redist.x64.exe. "+
					"Otherwise a sibling DLL (llama.dll, ggml*.dll) is missing from the same directory",
				path)
		}
		return 0, fmt.Errorf("LoadLibraryEx(%s) failed: %w", path, errno)
	}
	return h, nil
}
