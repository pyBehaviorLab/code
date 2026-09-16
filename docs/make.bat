@echo off
REM Minimal Sphinx build helper for Windows.
REM Usage: docs\make.bat html

if "%1"=="" (
  echo Usage: %0 html
  exit /b 2
)

set SPHINXBUILD=sphinx-build
set SOURCEDIR=%~dp0
set BUILDDIR=%~dp0_build

if "%1"=="html" (
  %SPHINXBUILD% -b html "%SOURCEDIR%" "%BUILDDIR%\\html"
  exit /b %ERRORLEVEL%
)

echo Unknown target: %1
exit /b 2

