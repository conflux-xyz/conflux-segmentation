import pytest

import numpy as np

from conflux_segmentation.utils import gaussian_weights, get_padding, sigmoid, softmax


@pytest.mark.parametrize(
    "size,tile_size,stride,expected",
    [
        # No padding needed
        pytest.param(512, 256, 256, (0, 0), id="exact_multiple"),
        pytest.param(1024, 512, 512, (0, 0), id="large_exact_multiple"),
        pytest.param(256, 256, 256, (0, 0), id="size_equals_tile"),
        pytest.param(256, 256, 128, (0, 0), id="overlap_no_padding"),
        # No overlapping but need padding
        pytest.param(500, 256, 256, (6, 6), id="even_padding"),
        pytest.param(900, 512, 512, (62, 62), id="large_even_padding"),
        pytest.param(1000, 256, 64, (12, 12), id="dense_overlap"),
        pytest.param(505, 256, 256, (3, 4), id="odd_padding"),
        pytest.param(901, 512, 512, (61, 62), id="large_odd_padding"),
        # With overlap
        pytest.param(200, 512, 256, (156, 156), id="small_even_padding"),
        pytest.param(500, 256, 128, (6, 6), id="overlap_no_padding"),
        pytest.param(201, 512, 256, (155, 156), id="small_odd_padding"),
        pytest.param(505, 256, 192, (67, 68), id="overlap_with_padding"),
        # Edge cases
        pytest.param(100, 256, 256, (78, 78), id="image_smaller_than_tile"),
    ],
)
def test_get_padding(size: int, tile_size: int, stride: int, expected: tuple[int, int]):
    assert get_padding(size, tile_size, stride) == expected


def test_gaussian_weights():
    size = 5
    weights = gaussian_weights(size)

    # Test basic properties
    assert weights.shape == (size, size)
    assert weights.dtype == np.float32
    assert np.isclose(weights.max(), 1.0)

    # Test symmetry
    assert np.allclose(weights, weights.T)  # Horizontal = Vertical
    assert np.allclose(weights[::-1, :], weights)  # Top = Bottom
    assert np.allclose(weights[:, ::-1], weights)  # Left = Right

    # Test with custom sigma
    weights_wide = gaussian_weights(size, sigma=size / 2)
    weights_narrow = gaussian_weights(size, sigma=size / 8)
    assert weights_wide[0, 0] > weights_narrow[0, 0]  # Wide sigma = slower falloff


def test_sigmoid():
    # Test basic properties
    x = np.array([0.0], dtype=np.float32)
    assert sigmoid(x).dtype == np.float32
    assert np.isclose(sigmoid(x)[0], 0.5)

    # Test array handling and range
    x = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
    y = sigmoid(x)
    assert y.shape == x.shape
    assert np.all(y >= 0) and np.all(y <= 1)
    assert np.isclose(y[0], 0.0, atol=1e-4)  # Very negative -> ~0
    assert np.isclose(y[1], 0.5)  # Zero -> 0.5
    assert np.isclose(y[2], 1.0, atol=1e-4)  # Very positive -> ~1

    # Test 2D array
    x_2d = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    y_2d = sigmoid(x_2d)
    assert y_2d.shape == x_2d.shape
    assert y_2d.dtype == np.float32


def test_softmax():
    # Test basic properties
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    softx = softmax(x)
    assert softx.dtype == np.float32
    assert np.allclose(softx, 1 / 3)
