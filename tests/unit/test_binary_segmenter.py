import pytest

import numpy as np
import numpy.typing as npt

from conflux_segmentation.binary_segmenter import BinarySegmenter
from conflux_segmentation.tile_segmenter import BinaryTileSegmenterBase


class MockTileSegmenter(BinaryTileSegmenterBase):
    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Return simple mock segmentation - center circle
        assert tiles.ndim == 4, (
            "Input image tiles must have 4 dimensions (N x C x H x W)"
        )
        batch_size = tiles.shape[0]
        assert tiles.shape[1] == 3, "Input image tiles must have 3 channels (RGB)"
        size = tiles.shape[2]
        masks = np.zeros((batch_size, 1, size, size), dtype=np.float32)
        y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
        mask = x * x + y * y <= (size // 4) ** 2
        masks[:] = mask
        return masks


def test_binary_segmenter_init():
    tile_segmenter = MockTileSegmenter()

    # Test default parameters
    segmenter = BinarySegmenter(tile_segmenter)
    assert segmenter.tile_size == 512
    assert segmenter.stride == round(512 * (1 - 0.125))
    assert segmenter.pad_value == 255

    # Test custom parameters
    segmenter = BinarySegmenter(
        tile_segmenter, tile_size=256, overlap=0.25, blend_mode="flat", pad_value=0
    )
    assert segmenter.tile_size == 256
    assert segmenter.stride == round(256 * (1 - 0.25))
    assert segmenter.pad_value == 0


def test_binary_segmenter_call():
    tile_segmenter = MockTileSegmenter()
    segmenter = BinarySegmenter(tile_segmenter, tile_size=64, overlap=0.5)

    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test segmentation
    mask = segmenter(image)

    # Check output properties
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)

    # Verify padding doesn't change output dimensions
    assert segmenter(np.zeros((50, 75, 3), dtype=np.uint8)).shape == (50, 75)


def test_invalid_input():
    tile_segmenter = MockTileSegmenter()
    segmenter = BinarySegmenter(tile_segmenter)

    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        segmenter(np.zeros((100, 100)))  # Missing channel dimension
