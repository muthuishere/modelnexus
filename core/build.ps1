# Build the modelnexus native bridge on Windows.
#
# The PowerShell counterpart to build.sh. It drives the same CMakeLists with the
# same two modes; only the platform mechanics differ -- .dll instead of .dylib/.so,
# a .zip release asset instead of .tar.gz, and no rpath (Windows resolves siblings
# from the DLL's own directory automatically, which is why CMakeLists sets rpath
# only on Apple/UNIX).
#
#   .\build.ps1                 prebuilt mode (default) -- download llama.cpp's
#                               official release libs, compile ONLY our bridge.
#   .\build.ps1 -Source         build llama.cpp from the checkout too. Slow.
#   .\build.ps1 -Clean          remove build\ and dist\ first.
#   .\build.ps1 -PrintTag       print the pinned llama.cpp tag and exit.
#
# Output: dist\<platform-key>\ containing llamabridge.dll plus the llama/ggml DLLs
# it needs at runtime -- self-contained, hand it to any FFI binding as-is.

[CmdletBinding()]
param(
    [switch]$Source,
    [switch]$Clean,
    [switch]$PrintTag
)

$ErrorActionPreference = 'Stop'

# Pinned llama.cpp release. Must match build.sh -- the tag is part of the release
# identity (ADR-0004), and two build scripts disagreeing about it would produce
# artifacts that claim the same version and behave differently.
$LlamaTag = if ($env:LLAMA_TAG) { $env:LLAMA_TAG } else { 'b9371' }

if ($PrintTag) { Write-Output $LlamaTag; exit 0 }

$Root = $PSScriptRoot
Push-Location $Root
try {
    if ($Clean) {
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$Root\build", "$Root\dist"
    }

    # ---- platform key -------------------------------------------------------
    $arch = switch ($env:PROCESSOR_ARCHITECTURE) {
        'AMD64' { 'x86_64' }
        'ARM64' { 'aarch64' }
        default { throw "unsupported arch: $env:PROCESSOR_ARCHITECTURE" }
    }
    $platform = "windows-$arch"
    $asset = switch ($platform) {
        'windows-x86_64'  { 'win-cpu-x64' }
        'windows-aarch64' { 'win-cpu-arm64' }
        default { throw "no prebuilt llama.cpp asset mapping for $platform" }
    }

    $headers  = Join-Path $Root 'third_party\llama.cpp'
    $buildDir = Join-Path $Root 'build\cmake'
    $dist     = Join-Path $Root "dist\$platform"
    $prebuilt = Join-Path $Root 'build\llama-prebuilt'
    $mode     = if ($Source) { 'source' } else { 'prebuilt' }

    Write-Host "==> modelnexus native bridge | platform=$platform mode=$mode llama.cpp=$LlamaTag"

    # ---- headers ------------------------------------------------------------
    # Needed in BOTH modes for include paths; never compiled in prebuilt mode.
    # Cloned on demand and gitignored, so the tag stays the single source of truth.
    if (-not (Test-Path (Join-Path $headers 'include\llama.h'))) {
        Write-Host "==> cloning llama.cpp $LlamaTag (shallow)"
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $headers
        git clone --depth 1 -b $LlamaTag https://github.com/ggml-org/llama.cpp $headers
        if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
    }

    New-Item -ItemType Directory -Force -Path $buildDir, $dist | Out-Null
    $cmakeArgs = @('-S', $Root, '-B', $buildDir, "-DLLB_HEADERS_DIR=$headers")

    if ($mode -eq 'prebuilt') {
        $archive = "llama-$LlamaTag-bin-$asset.zip"
        $url     = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/$archive"
        $extract = Join-Path $prebuilt $platform
        if (-not (Test-Path $extract)) {
            Write-Host "==> downloading $url"
            New-Item -ItemType Directory -Force -Path $extract, $prebuilt | Out-Null
            $zip = Join-Path $prebuilt $archive
            Invoke-WebRequest -Uri $url -OutFile $zip
            Expand-Archive -Path $zip -DestinationPath $extract -Force
        }
        # find_library needs the exact directory holding llama.lib / llama.dll.
        $llama = Get-ChildItem -Path $extract -Recurse -Filter 'llama.dll' | Select-Object -First 1
        if (-not $llama) { throw "prebuilt $archive did not contain llama.dll" }
        $libDir = $llama.DirectoryName
        Write-Host "==> linking against prebuilt libs in $libDir"
        $cmakeArgs += "-DLLB_PREBUILT_DIR=$libDir"
        $libSource = $libDir
    } else {
        Write-Host '==> building llama.cpp from source (this takes a while)'
        $libSource = $buildDir
    }

    # ---- build --------------------------------------------------------------
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }
    & cmake --build $buildDir --config Release
    if ($LASTEXITCODE -ne 0) { throw 'cmake build failed' }

    # ---- stage --------------------------------------------------------------
    $bridge = Get-ChildItem -Path $buildDir -Recurse -Filter 'llamabridge.dll' | Select-Object -First 1
    if (-not $bridge) { throw 'bridge not produced' }
    Copy-Item $bridge.FullName $dist -Force

    # Everything the bridge loads at runtime, side by side -- but not llama.cpp's
    # *-impl DLLs, which are tool implementations rather than runtime deps.
    Get-ChildItem -Path $libSource -Recurse -Filter '*.dll' |
        Where-Object { $_.Name -notlike '*-impl*' } |
        ForEach-Object { Copy-Item $_.FullName $dist -Force }

    $license = Join-Path $Root 'licenses\LICENSE-llama.cpp'
    if (Test-Path $license) { Copy-Item $license $dist -Force }

    Write-Host "==> staged to $dist"
    Get-ChildItem $dist | Select-Object -ExpandProperty Name
}
finally {
    Pop-Location
}
