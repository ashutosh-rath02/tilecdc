import os
import platform
import shutil
import sys

from setuptools import Extension, setup

if not sys.platform.startswith("linux") or platform.machine().lower() not in ("x86_64", "amd64"):
    sys.exit("tilecdc 0.1 supports x86-64 Linux only (it uses AMX/AVX-512 intrinsics and Linux arch_prctl).")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Vendor the C core so sdists build standalone.
VENDOR = os.path.join(HERE, "csrc")
if os.path.exists(os.path.join(ROOT, "src", "tilecdc.c")):
    os.makedirs(VENDOR, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "src", "tilecdc.c"), VENDOR)
    shutil.copy(os.path.join(ROOT, "include", "tilecdc.h"), VENDOR)

setup(
    ext_modules=[
        Extension(
            "tilecdc._tilecdc",
            sources=["_tilecdc.c", "csrc/tilecdc.c"],
            include_dirs=["csrc"],
            extra_compile_args=["-O3"],
        )
    ],
)
