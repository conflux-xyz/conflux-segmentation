import numpy as np
import pytest
from conflux_segmentation.transforms import (
    Normalize,
    ToFloat32,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class TestNormalize:
    """Tests for the Normalize transform class."""

    def test_init_default_values(self):
        """Test initialization with default mean, std, and max_pixel_value."""
        norm = Normalize()
        assert norm.dims == 3
        assert norm.mean.shape == (1, 3, 1, 1)
        assert norm.std.shape == (1, 3, 1, 1)
        assert norm.mean.dtype == np.float32
        assert norm.std.dtype == np.float32

    def test_init_custom_values(self):
        """Test initialization with custom mean and std."""
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.1, 0.1, 0.1)
        norm = Normalize(mean=custom_mean, std=custom_std)

        assert norm.dims == 3
        np.testing.assert_array_almost_equal(
            norm.mean.squeeze(), np.array(custom_mean, dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            norm.std.squeeze(), np.array(custom_std, dtype=np.float32)
        )

    def test_init_mismatched_mean_std_length(self):
        """Test that initialization fails when mean and std have different lengths."""
        with pytest.raises(
            AssertionError, match="Mean and std must have the same length"
        ):
            Normalize(mean=(0.5, 0.5), std=(0.1, 0.1, 0.1))

    def test_init_single_channel(self):
        """Test initialization with single channel (grayscale)."""
        norm = Normalize(mean=(0.5,), std=(0.1,))
        assert norm.dims == 1
        assert norm.mean.shape == (1, 1, 1, 1)
        assert norm.std.shape == (1, 1, 1, 1)

    def test_call_correct_input_shape(self):
        """Test that __call__ works with correct 4D input."""
        norm = Normalize()
        tiles = np.random.randint(0, 256, size=(2, 3, 64, 64), dtype=np.uint8)
        result = norm(tiles)

        assert result.shape == tiles.shape
        assert result.dtype == np.float32

    def test_call_output_dtype(self):
        """Test that output is always float32."""
        norm = Normalize()
        tiles = np.random.randint(0, 256, size=(1, 3, 32, 32), dtype=np.uint8)
        result = norm(tiles)

        assert result.dtype == np.float32

    def test_call_incorrect_ndim(self):
        """Test that __call__ fails with incorrect number of dimensions."""
        norm = Normalize()

        # 3D input
        tiles_3d = np.random.randint(0, 256, size=(3, 64, 64), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Input tiles must have 4 dimensions"):
            norm(tiles_3d)

        # 2D input
        tiles_2d = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Input tiles must have 4 dimensions"):
            norm(tiles_2d)

        # 5D input
        tiles_5d = np.random.randint(0, 256, size=(1, 3, 64, 64, 1), dtype=np.uint8)
        with pytest.raises(AssertionError, match="Input tiles must have 4 dimensions"):
            norm(tiles_5d)

    def test_call_incorrect_channels(self):
        """Test that __call__ fails when channel count doesn't match."""
        norm = Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1))

        # Wrong number of channels
        tiles_wrong_channels = np.random.randint(
            0, 256, size=(1, 4, 64, 64), dtype=np.uint8
        )
        with pytest.raises(AssertionError, match="Input tiles must have 3 channels"):
            norm(tiles_wrong_channels)

        # Single channel input with 3-channel normalizer
        tiles_single_channel = np.random.randint(
            0, 256, size=(1, 1, 64, 64), dtype=np.uint8
        )
        with pytest.raises(AssertionError, match="Input tiles must have 3 channels"):
            norm(tiles_single_channel)

    def test_call_normalization_values(self):
        """Test that normalization produces correct values."""
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        max_pixel_value = 1.0
        norm = Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value)

        # Create input with known values
        tiles = np.ones((1, 3, 2, 2), dtype=np.uint8)
        result = norm(tiles)

        # Expected: (1.0 / 1.0 - 0.5) / 0.5 = 1.0
        expected = np.ones((1, 3, 2, 2), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_with_zeros(self):
        """Test normalization with zero input values."""
        norm = Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0
        )
        tiles = np.zeros((1, 3, 4, 4), dtype=np.uint8)
        result = norm(tiles)

        assert result.shape == tiles.shape
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, np.zeros_like(result))

    def test_call_with_max_values(self):
        """Test normalization with maximum uint8 values."""
        norm = Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0
        )
        tiles = np.full((1, 3, 4, 4), 255, dtype=np.uint8)
        result = norm(tiles)

        expected = np.ones((1, 3, 4, 4), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_batch_processing(self):
        """Test normalization with multiple batches."""
        norm = Normalize()
        batch_sizes = [1, 2, 5, 10]

        for batch_size in batch_sizes:
            tiles = np.random.randint(
                0, 256, size=(batch_size, 3, 32, 32), dtype=np.uint8
            )
            result = norm(tiles)

            assert result.shape == (batch_size, 3, 32, 32)
            assert result.dtype == np.float32

    def test_call_preserves_spatial_dimensions(self):
        """Test that spatial dimensions are preserved."""
        norm = Normalize()
        spatial_dims = [(16, 16), (32, 64), (128, 256)]

        for h, w in spatial_dims:
            tiles = np.random.randint(0, 256, size=(1, 3, h, w), dtype=np.uint8)
            result = norm(tiles)

            assert result.shape == (1, 3, h, w)
            assert result.dtype == np.float32

    def test_imagenet_constants(self):
        """Test that IMAGENET constants are properly defined."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(isinstance(x, float) for x in IMAGENET_MEAN)
        assert all(isinstance(x, float) for x in IMAGENET_STD)

    def test_custom_max_pixel_value(self):
        """Test normalization with custom max_pixel_value."""
        # Test with max_pixel_value = 1.0 (already normalized input)
        norm = Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=1.0)
        tiles = np.ones((1, 3, 4, 4), dtype=np.uint8)
        result = norm(tiles)

        # Expected: (1 / 1.0 - 0.0) / 1.0 = 1.0
        np.testing.assert_array_almost_equal(result, np.ones_like(result))

    def test_grayscale_normalization(self):
        """Test normalization with single-channel grayscale images."""
        norm = Normalize(mean=(0.5,), std=(0.2,), max_pixel_value=255.0)
        tiles = np.random.randint(0, 256, size=(2, 1, 64, 64), dtype=np.uint8)
        result = norm(tiles)

        assert result.shape == (2, 1, 64, 64)
        assert result.dtype == np.float32


class TestToFloat32:
    """Tests for the ToFloat32 transform class."""

    def test_init_default_max_pixel_value(self):
        """Test initialization with default max_pixel_value."""
        transform = ToFloat32()
        assert transform.max_pixel_value == 255.0

    def test_init_custom_max_pixel_value(self):
        """Test initialization with custom max_pixel_value."""
        transform = ToFloat32(max_pixel_value=100.0)
        assert transform.max_pixel_value == 100.0

    def test_call_basic_conversion(self):
        """Test basic uint8 to float32 conversion."""
        transform = ToFloat32()
        tiles = np.array([[[[128]]]], dtype=np.uint8)
        result = transform(tiles)

        assert result.dtype == np.float32
        assert result.shape == tiles.shape
        np.testing.assert_almost_equal(result[0, 0, 0, 0], 128.0 / 255.0)

    def test_call_with_zeros(self):
        """Test conversion with zero input values."""
        transform = ToFloat32()
        tiles = np.zeros((1, 3, 4, 4), dtype=np.uint8)
        result = transform(tiles)

        assert result.shape == tiles.shape
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_call_with_max_values(self):
        """Test conversion with maximum uint8 values."""
        transform = ToFloat32()
        tiles = np.full((1, 3, 4, 4), 255, dtype=np.uint8)
        result = transform(tiles)

        assert result.dtype == np.float32
        expected = np.ones((1, 3, 4, 4), dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_with_various_shapes(self):
        """Test conversion with various input shapes."""
        transform = ToFloat32()
        shapes = [(1, 3, 32, 32), (2, 3, 64, 64), (5, 1, 16, 16), (10, 4, 128, 256)]

        for shape in shapes:
            tiles = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            result = transform(tiles)

            assert result.shape == shape
            assert result.dtype == np.float32

    def test_call_output_range(self):
        """Test that output values are in the correct range [0, 1]."""
        transform = ToFloat32()
        tiles = np.random.randint(0, 256, size=(2, 3, 32, 32), dtype=np.uint8)
        result = transform(tiles)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_call_custom_max_pixel_value(self):
        """Test conversion with custom max_pixel_value."""
        transform = ToFloat32(max_pixel_value=100.0)
        tiles = np.array([[[[100]]]], dtype=np.uint8)
        result = transform(tiles)

        assert result.dtype == np.float32
        np.testing.assert_almost_equal(result[0, 0, 0, 0], 1.0)

    def test_call_specific_values(self):
        """Test conversion with specific known values."""
        transform = ToFloat32(max_pixel_value=255.0)
        tiles = np.array([[[[0, 127, 255]]]], dtype=np.uint8)
        result = transform(tiles)

        expected = np.array([[[[0.0, 127.0 / 255.0, 1.0]]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_call_preserves_spatial_dimensions(self):
        """Test that spatial dimensions are preserved."""
        transform = ToFloat32()
        spatial_dims = [(16, 16), (32, 64), (128, 256)]

        for h, w in spatial_dims:
            tiles = np.random.randint(0, 256, size=(1, 3, h, w), dtype=np.uint8)
            result = transform(tiles)

            assert result.shape == (1, 3, h, w)
            assert result.dtype == np.float32

    def test_call_batch_processing(self):
        """Test conversion with multiple batches."""
        transform = ToFloat32()
        batch_sizes = [1, 2, 5, 10]

        for batch_size in batch_sizes:
            tiles = np.random.randint(
                0, 256, size=(batch_size, 3, 32, 32), dtype=np.uint8
            )
            result = transform(tiles)

            assert result.shape == (batch_size, 3, 32, 32)
            assert result.dtype == np.float32

    def test_call_single_channel(self):
        """Test conversion with single-channel grayscale images."""
        transform = ToFloat32()
        tiles = np.random.randint(0, 256, size=(2, 1, 64, 64), dtype=np.uint8)
        result = transform(tiles)

        assert result.shape == (2, 1, 64, 64)
        assert result.dtype == np.float32

    def test_call_multi_channel(self):
        """Test conversion with various channel counts."""
        transform = ToFloat32()
        channel_counts = [1, 3, 4]

        for channels in channel_counts:
            tiles = np.random.randint(
                0, 256, size=(1, channels, 32, 32), dtype=np.uint8
            )
            result = transform(tiles)

            assert result.shape == (1, channels, 32, 32)
            assert result.dtype == np.float32
