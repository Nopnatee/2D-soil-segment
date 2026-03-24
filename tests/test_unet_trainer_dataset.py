from __future__ import annotations

import numpy as np
from PIL import Image

from soil_segment.unet_trainer import BeadDataset, load_image_mask_pair


def test_bead_dataset_resolves_roboflow_mask_names(tmp_path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image_path = image_dir / "sample.rf.123.jpg"
    mask_path = mask_dir / "sample.rf.123_mask.png"

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.full((8, 8), 3, dtype=np.uint8)).save(mask_path)

    dataset = BeadDataset(str(image_dir), str(mask_dir))

    image, mask = dataset[0]
    raw_image, raw_mask = load_image_mask_pair(dataset, 0)

    assert image.size == (8, 8)
    assert mask.mode == "L"
    assert np.array(mask).tolist() == np.full((8, 8), 3, dtype=np.uint8).tolist()
    assert raw_image.size == (8, 8)
    assert raw_mask.mode == "L"
