"""Compatibility shim: forwards to the package module.

Prefer using `python -m soil_segment.visualizer` or console script.
"""
from soil_segment.visualizer import *  # re-export

if __name__ == "__main__":
    try:
        from soil_segment.visualizer import main as _main
        _main()
    except Exception as _e:
        raise

