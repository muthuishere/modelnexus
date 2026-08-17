//go:build windows

package modelnexus

import "unsafe"

func unsafePointerOf(p *uint16) unsafe.Pointer { return unsafe.Pointer(p) }
