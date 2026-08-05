"""Setuptools bridge for the selectable experimental build profile."""

from pathlib import Path
import sys

from setuptools import setup


sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_support.packaging import build_configuration


setup(**build_configuration().setup_arguments())
