"""Backward-compatibility shim after src/ layout migration.

Allows legacy imports `from custom_unet import ...` to work whether or not
the package has been installed. If `soil_segment` isn't importable, this
shim prepends the local `src` directory to `sys.path` and retries.
"""

from typing import Tuple

try:
    from soil_segment.custom_unet import ConvBlock, SimpleUNet
except ModuleNotFoundError:
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Retry import after adjusting sys.path
    from soil_segment.custom_unet import ConvBlock, SimpleUNet  # type: ignore

__all__ = ["ConvBlock", "SimpleUNet"]
