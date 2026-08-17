@echo off
rem Build the modelnexus native bridge on Windows.
rem
rem The batch counterpart to build.sh, driving the same CMakeLists. Plain cmd rather
rem than PowerShell: execution policy never blocks a .cmd, and Windows 10 1803+ ships
rem both curl.exe and tar.exe (bsdtar, which reads zip), so nothing needs installing.
rem
rem   build.cmd                SOURCE mode -- the default here, unlike build.sh.
rem   build.cmd --prebuilt     link llama.cpp's release binaries. DOES NOT WORK; see below.
rem   build.cmd --clean        remove build\ and dist\ first.
rem   build.cmd --print-tag    print the pinned llama.cpp tag and exit.
rem
rem WHY WINDOWS DEFAULTS TO SOURCE WHEN UNIX DEFAULTS TO PREBUILT
rem
rem llama.cpp's Windows release archive ships 29 DLLs and ZERO .lib import libraries.
rem MSVC cannot link a DLL without its import library, so CMake fails with
rem   "Could not find LLB_LIB_LLAMA using the following names: llama"
rem -- which is exactly what happened the first time this ran in CI. Nothing in this
rem repo can fix that; it is what upstream publishes.
rem
rem Building llama.cpp from source produces the .lib files as a matter of course. It
rem is slow, and that is acceptable: this runs only in the rare tier-1 natives
rem workflow (ADR-0004), never on a release tag.
rem
rem Output: dist\<platform-key>\ with llamabridge.dll and the llama/ggml DLLs it loads
rem at runtime. Windows resolves siblings from the DLL's own directory, which is why
rem CMakeLists sets rpath only on Apple/UNIX -- nothing extra is needed here.

setlocal EnableDelayedExpansion

rem Pinned llama.cpp release. MUST match build.sh: the tag is part of the release
rem identity (ADR-0004), and two build scripts disagreeing about it would produce
rem artifacts that claim one version and behave like another.
if "%LLAMA_TAG%"=="" set "LLAMA_TAG=b9371"

rem Source by default here -- see the note above. --prebuilt is kept so the failure is
rem reproducible and explicit rather than mysterious.
set "MODE=source"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="--source"    set "MODE=source"      & shift & goto parse
if /I "%~1"=="--prebuilt"  set "MODE=prebuilt"    & shift & goto parse
if /I "%~1"=="--clean"     set "DO_CLEAN=1"       & shift & goto parse
if /I "%~1"=="--print-tag" echo %LLAMA_TAG%       & exit /b 0
if /I "%~1"=="--help"      goto usage
if /I "%~1"=="-h"          goto usage
echo unknown flag: %~1 ^(try --help^)>&2
exit /b 2
:parsed

if defined DO_CLEAN (
  if exist "%ROOT%\build" rmdir /s /q "%ROOT%\build"
  if exist "%ROOT%\dist"  rmdir /s /q "%ROOT%\dist"
)

rem ---- platform key ---------------------------------------------------------
rem PROCESSOR_ARCHITECTURE is the *process* arch. A 32-bit shell on 64-bit Windows
rem reports x86 and sets PROCESSOR_ARCHITEW6432 instead -- honour that, or a build
rem on a perfectly capable machine claims an unsupported platform.
set "ARCHRAW=%PROCESSOR_ARCHITECTURE%"
if not "%PROCESSOR_ARCHITEW6432%"=="" set "ARCHRAW=%PROCESSOR_ARCHITEW6432%"

if /I "%ARCHRAW%"=="AMD64" (
  set "ARCH=x86_64"
  set "ASSET=win-cpu-x64"
) else if /I "%ARCHRAW%"=="ARM64" (
  set "ARCH=aarch64"
  set "ASSET=win-cpu-arm64"
) else (
  echo unsupported arch: %ARCHRAW%>&2
  exit /b 1
)
set "PLATFORM=windows-%ARCH%"

set "HEADERS=%ROOT%\third_party\llama.cpp"
set "BUILDDIR=%ROOT%\build\cmake"
set "DIST=%ROOT%\dist\%PLATFORM%"
set "PREBUILT=%ROOT%\build\llama-prebuilt"

echo ==^> modelnexus native bridge ^| platform=%PLATFORM% mode=%MODE% llama.cpp=%LLAMA_TAG%

rem ---- headers --------------------------------------------------------------
rem Needed in BOTH modes for include paths; never compiled in prebuilt mode. Cloned
rem on demand and gitignored, so the pinned tag stays the single source of truth.
if not exist "%HEADERS%\include\llama.h" (
  echo ==^> cloning llama.cpp %LLAMA_TAG% ^(shallow^)
  if exist "%HEADERS%" rmdir /s /q "%HEADERS%"
  git clone --depth 1 -b %LLAMA_TAG% https://github.com/ggml-org/llama.cpp "%HEADERS%"
  if errorlevel 1 echo git clone failed>&2 & exit /b 1
)

if not exist "%BUILDDIR%" mkdir "%BUILDDIR%"
if not exist "%DIST%"     mkdir "%DIST%"

set "CMAKEARGS=-S "%ROOT%" -B "%BUILDDIR%" -DLLB_HEADERS_DIR="%HEADERS%""

rem On ARM64, build with clang-cl instead of MSVC.
rem
rem Not a preference -- llama.cpp refuses outright:
rem   ggml/src/ggml-cpu/CMakeLists.txt:106  "MSVC is not supported for ARM, use clang"
rem Its ARM CPU kernels use GCC/clang vector intrinsics MSVC does not implement, so
rem there is no MSVC path to fall back to. -T ClangCL selects the clang-cl toolset
rem that ships with Visual Studio; CMAKE_MSVC_RUNTIME_LIBRARY still applies, so the
rem static-CRT choice carries over and the DLL stays self-contained.
if /I "%ARCH%"=="aarch64" (
  echo ==^> ARM64: using the ClangCL toolset ^(llama.cpp rejects MSVC here^)
  set "CMAKEARGS=!CMAKEARGS! -T ClangCL"
)

if /I "%MODE%"=="prebuilt" (
  set "ARCHIVE=llama-%LLAMA_TAG%-bin-%ASSET%.zip"
  set "URL=https://github.com/ggml-org/llama.cpp/releases/download/%LLAMA_TAG%/!ARCHIVE!"
  set "EXTRACT=%PREBUILT%\%PLATFORM%"
  if not exist "!EXTRACT!" (
    echo ==^> downloading !URL!
    if not exist "%PREBUILT%" mkdir "%PREBUILT%"
    mkdir "!EXTRACT!"
    curl -fsSL "!URL!" -o "%PREBUILT%\!ARCHIVE!"
    if errorlevel 1 echo download failed>&2 & exit /b 1
    rem tar.exe is bsdtar on Windows and reads zip natively -- no unzip needed.
    tar -xf "%PREBUILT%\!ARCHIVE!" -C "!EXTRACT!"
    if errorlevel 1 echo extract failed>&2 & exit /b 1
  )
  rem find_library needs the exact directory holding llama.lib / llama.dll.
  set "LIBDIR="
  for /f "delims=" %%F in ('dir /s /b "!EXTRACT!\llama.dll" 2^>nul') do (
    if not defined LIBDIR set "LIBDIR=%%~dpF"
  )
  if not defined LIBDIR echo prebuilt !ARCHIVE! did not contain llama.dll>&2 & exit /b 1
  echo WARNING: prebuilt mode has no .lib import libraries upstream and will fail to>&2
  echo          configure. Use source mode ^(the default^) instead.>&2
  if "!LIBDIR:~-1!"=="\" set "LIBDIR=!LIBDIR:~0,-1!"
  echo ==^> linking against prebuilt libs in !LIBDIR!
  set "CMAKEARGS=!CMAKEARGS! -DLLB_PREBUILT_DIR="!LIBDIR!""
  set "LIBSOURCE=!LIBDIR!"
) else (
  echo ==^> building llama.cpp from source ^(this takes a while^)
  set "LIBSOURCE=%BUILDDIR%"
)

rem ---- build ----------------------------------------------------------------
cmake %CMAKEARGS%
if errorlevel 1 echo cmake configure failed>&2 & exit /b 1
cmake --build "%BUILDDIR%" --config Release
if errorlevel 1 echo cmake build failed>&2 & exit /b 1

rem ---- stage ----------------------------------------------------------------
set "BRIDGE="
for /f "delims=" %%F in ('dir /s /b "%BUILDDIR%\llamabridge.dll" 2^>nul') do (
  if not defined BRIDGE set "BRIDGE=%%F"
)
if not defined BRIDGE echo bridge not produced>&2 & exit /b 1
copy /y "%BRIDGE%" "%DIST%\" >nul

rem Everything the bridge loads at runtime, side by side -- but not llama.cpp's
rem *-impl DLLs, which are tool implementations rather than runtime dependencies of
rem the bridge, and would roughly double dist\ for nothing.
for /f "delims=" %%F in ('dir /s /b "%LIBSOURCE%\*.dll" 2^>nul') do (
  echo %%~nxF | findstr /i /c:"-impl" >nul || copy /y "%%F" "%DIST%\" >nul
)

rem Every notice, not just llama.cpp's -- nlohmann/json is header-only and compiled
rem into the bridge, so its MIT notice travels with the binary too.
if exist "%ROOT%\licenses\" copy /y "%ROOT%\licenses\*" "%DIST%\" >nul

echo ==^> staged to %DIST%
dir /b "%DIST%"
exit /b 0

:usage
echo Build the modelnexus native bridge on Windows.
echo.
echo   build.cmd                source mode ^(default^) -- builds llama.cpp too. Slow,
echo                            and necessary: upstream ships no Windows import libs.
echo   build.cmd --prebuilt     link release binaries. Fails; kept for diagnosis.
echo   build.cmd --clean        remove build\ and dist\ first.
echo   build.cmd --print-tag    print the pinned llama.cpp tag and exit.
exit /b 0
