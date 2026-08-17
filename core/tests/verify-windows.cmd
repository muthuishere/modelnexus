@echo off
rem Verify the PUBLISHED Windows native actually works, on real Windows.
rem
rem WHY THIS EXISTS. natives.yml builds windows-x86_64 in CI and stages it, and
rem CI going green proves it COMPILED. Nothing has ever proved it LOADS and
rem RUNS. Those are different claims, and today's release shipped a Windows
rem asset on the strength of the weaker one. PUBLISHING.md has said "Windows
rem never run outside CI" since 0.1.0; this is how that line gets deleted.
rem
rem Batch, not PowerShell (owner rule): execution policy never blocks a .cmd,
rem and Windows 10 1803+ ships curl.exe and tar.exe (bsdtar, reads zip).
rem
rem   verify-windows.cmd                     download the released native, verify
rem   verify-windows.cmd C:\path\model.gguf  ...and run real inference too
rem
rem Run it from inside the Parallels VM. The Mac home is mounted at \\Mac\Home,
rem so this file is reachable without copying anything.

setlocal EnableDelayedExpansion

set "LLAMA_TAG=b9371"
set "ASSET=natives-windows-x86_64.zip"
set "URL=https://github.com/muthuishere/modelnexus/releases/download/natives-%LLAMA_TAG%/%ASSET%"
set "WORK=%TEMP%\modelnexus-winverify"
set "MODEL=%~1"

echo ==^> modelnexus Windows verification
echo     tag:   %LLAMA_TAG%
echo     work:  %WORK%
echo.

rem ---- find the repo, whether run from the share or a clone ----------------
set "HERE=%~dp0"
for %%I in ("%HERE%..\..") do set "ROOT=%%~fI"
if not exist "%ROOT%\bindings\python\modelnexus\__init__.py" (
  echo FAIL: cannot find the repo from %HERE%
  echo       expected %ROOT%\bindings\python\modelnexus
  exit /b 1
)
echo     repo:  %ROOT%

rem ---- python is the only runtime needed --------------------------------
rem The Python binding is pure ctypes: no compiler, no toolchain, no wheel to
rem build. That makes it the cheapest honest way to exercise a native DLL on a
rem machine that has nothing else installed.
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (where python >nul 2>&1 && set "PY=python")
if not defined PY (
  echo FAIL: no Python found. Install it from the Store or python.org, then re-run.
  echo       ^(The binding is ctypes-only, so nothing else is needed.^)
  exit /b 1
)
for /f "delims=" %%V in ('%PY% -c "import sys;print(sys.version.split()[0])" 2^>nul') do set "PYVER=%%V"
echo     python: %PYVER%
echo.

rem ---- fetch the RELEASED asset, not a local build ------------------------
rem Deliberately the published artifact. A local build proves the source
rem compiles here; only the released zip proves what a user downloads works.
if not exist "%WORK%" mkdir "%WORK%"
if not exist "%WORK%\%ASSET%" (
  echo ==^> downloading %URL%
  curl -fsSL "%URL%" -o "%WORK%\%ASSET%"
  if errorlevel 1 echo FAIL: download failed & exit /b 1
)
if exist "%WORK%\windows-x86_64" rmdir /s /q "%WORK%\windows-x86_64"
tar -xf "%WORK%\%ASSET%" -C "%WORK%"
if errorlevel 1 echo FAIL: extract failed ^(is it a real zip?^) & exit /b 1

set "LIBDIR=%WORK%\windows-x86_64"
if not exist "%LIBDIR%\llamabridge.dll" (
  echo FAIL: the asset contains no llamabridge.dll
  dir /b "%WORK%"
  exit /b 1
)
echo ==^> extracted to %LIBDIR%
dir /b "%LIBDIR%\*.dll"
echo.

rem ---- the actual test ----------------------------------------------------
set "MODELNEXUS_LIB=%LIBDIR%"
set "PYTHONPATH=%ROOT%\bindings\python"
set "MODELNEXUS_MODEL=%MODEL%"

%PY% "%ROOT%\core\tests\verify_windows.py"
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo ==^> WINDOWS VERIFIED
) else (
  echo ==^> WINDOWS FAILED ^(exit %RC%^) -- read the output above
)
exit /b %RC%
