#!/usr/bin/env python3
"""Build the optional Cython TAAS extension beside infer_offline.py."""

from pathlib import Path
import os

import numpy
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as error:
    raise SystemExit(
        "Cython is required. Install it with: python -m pip install cython"
    ) from error


HERE = Path(__file__).resolve().parent
os.chdir(HERE)

extensions = [
    Extension(
        "tass_cython_distance",
        [str(HERE / "tass_cython_distance.pyx")],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="advis-taas-cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
