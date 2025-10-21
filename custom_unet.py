# Backward-compatibility shim after src/ layout migration
from soil_segment.custom_unet import ConvBlock, SimpleUNet

__all__ = ["ConvBlock", "SimpleUNet"]
