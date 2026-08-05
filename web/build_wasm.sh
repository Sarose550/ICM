#!/usr/bin/env bash
# Compiles the ICM library to WebAssembly against the generic device profile
# (uncalibrated fallback: correct results, FFTW_ESTIMATE plans, always-hybrid
# B=32). Requires web/build_fftw.sh to have been run first.
# Output: web/dist/icm.mjs + web/dist/icm.wasm.
set -euo pipefail

cd "$(dirname "$0")"
FFTW_VERSION=3.3.10

test -f .build/libfftw3-wasm.a || { echo "run web/build_fftw.sh first" >&2; exit 1; }
mkdir -p dist

emcc -O3 \
    -I../src/cpu \
    -I../devices/generic \
    -I".build/fftw-${FFTW_VERSION}/api" \
    ../src/cpu/icm.c \
    .build/libfftw3-wasm.a \
    -sMODULARIZE=1 \
    -sEXPORT_ES6=1 \
    -sEXPORT_NAME=createIcmModule \
    -sENVIRONMENT=web,worker,node \
    -sALLOW_MEMORY_GROWTH=1 \
    -sINITIAL_MEMORY=64MB \
    -sSTACK_SIZE=5MB \
    -sEXPORTED_FUNCTIONS=_icm_init,_icm_equity,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=HEAPF64 \
    -o dist/icm.mjs

echo "OK: web/dist/icm.mjs + web/dist/icm.wasm"
