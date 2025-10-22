"""Compatibility shim: forwards to the package module.

Prefer importing and running from `soil_segment.trainer`.
"""
from soil_segment.trainer import *  # re-export

if __name__ == "__main__":
    try:
        from soil_segment.trainer import main as _main
        _main()
    except Exception as _e:
        # If no main() exists or runtime error occurs, surface it
        raise

