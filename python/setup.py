"""
Setup script for the ICM Python package.

Builds the ICM shared library (libicm.so / libicm.dylib) as part of
the install process, then installs the ctypes-based Python bindings.

Prerequisites:
  - C compiler (gcc or clang)
  - FFTW3 library (libfftw3-dev on Linux, `brew install fftw` on macOS)
  - On macOS: Accelerate framework (included with Xcode)

Usage:
  pip install .              # install from the python/ directory
  pip install -e .           # editable install (for development)
  pip install .[test]        # install with test dependencies
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


class BuildSharedLib(build_ext):
    """Custom build step that compiles icm.c into a shared library."""

    def run(self):
        # Project root is one level above python/
        project_root = Path(__file__).resolve().parent.parent
        src_file = project_root / "src" / "icm.c"
        build_dir = project_root / "build"
        build_dir.mkdir(exist_ok=True)

        system = platform.system()
        if system == "Darwin":
            lib_name = "libicm.dylib"
        else:
            lib_name = "libicm.so"

        lib_path = build_dir / lib_name

        # Determine device (default m3_max on macOS, zen4 on Linux)
        device = os.environ.get("ICM_DEVICE")
        if device is None:
            device = "m3_max" if system == "Darwin" else "zen4"

        device_dir = project_root / "devices" / device

        cc = os.environ.get("CC", "gcc")
        cflags = ["-O3", "-march=native", "-Wall", "-fPIC", "-shared"]
        includes = [
            f"-I{project_root / 'src'}",
            f"-I{device_dir}",
        ]
        ldflags = ["-lfftw3", "-lm"]

        if system == "Darwin":
            includes.append("-I/opt/homebrew/include")
            ldflags = ["-L/opt/homebrew/lib"] + ldflags + ["-framework", "Accelerate"]
        else:
            ldflags.append("-ldl")
            # Auto-detect AOCL-FFTW
            aocl_path = Path("/usr/local/aocl-fftw/lib/libfftw3.so")
            if aocl_path.exists():
                includes.append("-I/usr/local/aocl-fftw/include")
                ldflags = [
                    "-L/usr/local/aocl-fftw/lib",
                    "-Wl,-rpath,/usr/local/aocl-fftw/lib",
                ] + ldflags

        cmd = [cc] + cflags + includes + [str(src_file), "-o", str(lib_path)] + ldflags
        print(f"Building shared library: {' '.join(cmd)}")

        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Failed to build ICM shared library: {e}", file=sys.stderr)
            raise

        print(f"Built: {lib_path}")

        # Also run parent build_ext (no-op since we have no ext_modules)
        super().run()


class CustomDist(Distribution):
    """Ensure build_ext runs even without ext_modules."""

    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    setup(
        cmdclass={"build_ext": BuildSharedLib},
        distclass=CustomDist,
    )
