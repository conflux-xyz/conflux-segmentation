import pytest

import numpy as np
import numpy.typing as npt

from conflux_segmentation.segmenter import Segmenter
from conflux_segmentation.tile_segmenter import TileSegmenterBase


class MockTileSegmenter(TileSegmenterBase):
    def __init__(self, num_classes: int = 1, normalize_probs: bool = False):
        super().__init__()
        self.num_classes = num_classes
        # True for multiclass, False for binary or multilabel
        self.normalize_probs = normalize_probs

    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # Return simple mock segmentation - center circle
        assert tiles.ndim == 4, (
            "Input image tiles must have 4 dimensions (N x C x H x W)"
        )
        batch_size = tiles.shape[0]
        assert tiles.shape[1] == 3, "Input image tiles must have 3 channels (RGB)"
        size = tiles.shape[2]
        probabilities = np.random.rand(batch_size, self.num_classes, size, size).astype(
            np.float32
        )
        if self.normalize_probs:
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def test_binary_segmenter_init():
    tile_segmenter = MockTileSegmenter()

    # Test default parameters
    segmenter = Segmenter(tile_segmenter)
    assert segmenter.tile_size == 512
    assert segmenter.stride == round(512 * (1 - 0.125))
    assert segmenter.pad_value == 255

    # Test custom parameters
    segmenter = Segmenter(
        tile_segmenter, tile_size=256, overlap=0.25, blend_mode="flat", pad_value=0
    )
    assert segmenter.tile_size == 256
    assert segmenter.stride == round(256 * (1 - 0.25))
    assert segmenter.pad_value == 0


def test_binary_segmenter_call():
    tile_segmenter = MockTileSegmenter()
    segmenter = Segmenter(tile_segmenter, tile_size=64, overlap=0.5, batch_size=2)

    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test segmentation
    result = segmenter(image).to_binary()
    mask = result.get_mask()

    # Check output properties
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100)

    # Verify padding doesn't change output dimensions
    image = np.zeros((50, 75, 3), dtype=np.uint8)
    result = segmenter(image).to_binary()
    mask = result.get_mask()
    assert mask.shape == (50, 75)


def test_binary_invalid_input():
    tile_segmenter = MockTileSegmenter()
    segmenter = Segmenter(tile_segmenter)

    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        segmenter(np.zeros((100, 100)))  # Missing channel dimension


def test_multiclass_segmenter_init():
    tile_segmenter = MockTileSegmenter(5, normalize_probs=True)

    # Test default parameters
    segmenter = Segmenter(tile_segmenter, num_classes=5)
    assert segmenter.tile_size == 512
    assert segmenter.stride == round(512 * (1 - 0.125))
    assert segmenter.pad_value == 255

    # Test custom parameters
    segmenter = Segmenter(
        tile_segmenter,
        num_classes=5,
        tile_size=256,
        overlap=0.25,
        blend_mode="flat",
        pad_value=0,
    )
    assert segmenter.tile_size == 256
    assert segmenter.stride == round(256 * (1 - 0.25))
    assert segmenter.pad_value == 0


def test_multiclass_segmenter_call():
    tile_segmenter = MockTileSegmenter(5, normalize_probs=True)
    segmenter = Segmenter(tile_segmenter, num_classes=5, tile_size=64, overlap=0.5)

    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test segmentation
    result = segmenter(image).to_multiclass()
    mask = result.get_mask()
    mask_proba = result.get_mask_proba()

    # Check output properties
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint
    assert mask.shape == (100, 100)
    assert isinstance(mask_proba, np.ndarray)
    assert mask_proba.dtype == np.float32
    assert mask_proba.shape == (100, 100, 5)
    assert np.allclose(mask_proba.sum(axis=-1), 1)

    # Verify padding doesn't change output dimensions
    image = np.zeros((50, 75, 3), dtype=np.uint8)
    result = segmenter(image).to_multiclass()
    mask = result.get_mask()
    mask_proba = result.get_mask_proba()
    assert mask.shape == (50, 75)
    assert mask_proba.shape == (50, 75, 5)
    assert np.allclose(mask_proba.sum(axis=-1), 1)


def test_multiclass_invalid_input():
    tile_segmenter = MockTileSegmenter(5, normalize_probs=True)
    segmenter = Segmenter(tile_segmenter, num_classes=5)

    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        segmenter(np.zeros((100, 100)))  # Missing channel dimension


def test_multilabel_segmenter_init():
    tile_segmenter = MockTileSegmenter(5)

    # Test default parameters
    segmenter = Segmenter(tile_segmenter, num_classes=5)
    assert segmenter.tile_size == 512
    assert segmenter.stride == round(512 * (1 - 0.125))
    assert segmenter.pad_value == 255

    # Test custom parameters
    segmenter = Segmenter(
        tile_segmenter,
        num_classes=5,
        tile_size=256,
        overlap=0.25,
        blend_mode="flat",
        pad_value=0,
    )
    assert segmenter.tile_size == 256
    assert segmenter.stride == round(256 * (1 - 0.25))
    assert segmenter.pad_value == 0


def test_multilabel_segmenter_call():
    tile_segmenter = MockTileSegmenter(5)
    segmenter = Segmenter(tile_segmenter, num_classes=5, tile_size=64, overlap=0.5)

    # Create test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Test segmentation
    result = segmenter(image).to_multilabel()
    mask = result.get_mask()
    mask_proba = result.get_mask_proba()

    # Check output properties
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool_
    assert mask.shape == (100, 100, 5)
    assert isinstance(mask_proba, np.ndarray)
    assert mask_proba.dtype == np.float32
    assert mask_proba.shape == (100, 100, 5)
    assert np.all((mask_proba > 0) & (mask_proba < 1))

    # Verify padding doesn't change output dimensions
    image = np.zeros((50, 75, 3), dtype=np.uint8)
    result = segmenter(image).to_multilabel()
    mask = result.get_mask()
    mask_proba = result.get_mask_proba()
    assert mask.shape == (50, 75, 5)
    assert mask_proba.shape == (50, 75, 5)
    assert np.all((mask_proba > 0) & (mask_proba < 1))


def test_multilabel_invalid_input():
    tile_segmenter = MockTileSegmenter(5)
    segmenter = Segmenter(tile_segmenter, num_classes=5)

    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        segmenter(np.zeros((100, 100)))  # Missing channel dimension
