"""Puts this repository's topic directories on the import path.

The modules here are scripts rather than a package: they are run directly and they import
each other by bare name. They are grouped into topic directories so that the repository can
be read, and this restores the flat import namespace that grouping takes away. Importing it
has no other effect, and a module at the root does not need it.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DIRS = (
    "anchors",
    "audio",
    "benchmark",
    "capture",
    "census",
    "corpus",
    "judges",
    "record",
    "sittings",
)

for _d in DIRS:
    _p = os.path.join(ROOT, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
