#!/usr/bin/env bash
# Builds FFTW as a WebAssembly static library with Emscripten.
# Output: web/.build/libfftw3-wasm.a plus headers under web/.build/fftw-<ver>/api.
# The FFTW version is pinned and checksum-verified; the distributed widget
# bundle is a GPL combined work because of this dependency (see
# web/LICENSE-THIRD-PARTY.md).
set -euo pipefail

FFTW_VERSION=3.3.10
FFTW_SHA256=56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467

cd "$(dirname "$0")"
mkdir -p .build
cd .build

if [ ! -f "fftw-${FFTW_VERSION}.tar.gz" ]; then
    curl -fLO "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
fi
echo "${FFTW_SHA256}  fftw-${FFTW_VERSION}.tar.gz" | shasum -a 256 -c -

rm -rf "fftw-${FFTW_VERSION}"
tar xzf "fftw-${FFTW_VERSION}.tar.gz"
cd "fftw-${FFTW_VERSION}"

# FFTW 3.3.10 ships a config.sub that predates the emscripten triple.
cp ../../build-support/config.sub ../../build-support/config.guess .

emconfigure ./configure \
    --host=wasm32-unknown-emscripten \
    --disable-fortran \
    --disable-shared \
    --enable-static \
    --disable-doc \
    CFLAGS="-O3"
emmake make -j"$(getconf _NPROCESSORS_ONLN)"

cp .libs/libfftw3.a ../libfftw3-wasm.a
echo "OK: web/.build/libfftw3-wasm.a"
