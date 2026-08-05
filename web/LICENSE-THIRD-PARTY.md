# Third-Party Licenses (WebAssembly Bundle)

The page served at https://sarose550.github.io/ICM/ includes a prebuilt
WebAssembly bundle (`web/dist/icm.mjs` and `web/dist/icm.wasm`) that combines
two works:

1. The ICM equity computation library (`src/cpu/`), copyright the ICM
   contributors, licensed under the MIT license (see the repository root
   [LICENSE](../LICENSE)).

2. FFTW 3.3.10, copyright Matteo Frigo and the Massachusetts Institute of
   Technology, licensed under the GNU General Public License version 2 or
   any later version (GPL-2.0-or-later). FFTW is available from
   https://www.fftw.org.

Because the WebAssembly bundle links these two works into a single
executable, the combined work is distributed under the terms of the GPL as
a combined work (GPL-2.0-or-later).

## Corresponding Source

The complete corresponding source for the combined work consists of:

- This repository (https://github.com/Sarose550/ICM), which includes the
  ICM library (MIT) and the build scripts that fetch and compile FFTW.
- The FFTW 3.3.10 source tarball, pinned by `web/build_fftw.sh` with
  SHA-256:
  `56c932549852cddcfafdab3820b0200c7742675be92179e59e6215b340e26467`

The build scripts `web/build_fftw.sh` and `web/build_wasm.sh` reproduce the
bundle from these sources.

## Rebuilding

Install the Emscripten compiler (`emcc` on your PATH), then run:

```bash
web/build_fftw.sh
web/build_wasm.sh
```

The first script downloads the pinned FFTW tarball, verifies its SHA-256
checksum, and compiles FFTW as a WebAssembly static library. The second
script links the ICM library against that FFTW build and emits the bundle
into `web/dist/`.

## No Warranty

THE COMBINED WORK IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, TO THE
EXTENT PERMITTED BY LAW, AS SET FORTH IN THE GPL.
