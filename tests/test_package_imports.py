from __future__ import annotations

import importlib

import pytest


def test_simple_unet_forward_pass() -> None:
    """Ensure the core model can run a forward pass with dummy inputs."""
    torch = pytest.importorskip("torch")
    module = importlib.import_module("soil_segment.custom_unet")
    simple_unet = getattr(module, "SimpleUNet")

    model = simple_unet(in_channels=3, n_classes=2)
    dummy = torch.randn(1, 3, 64, 64)
    output = model(dummy)

    assert output.shape == (1, 2, 64, 64)
