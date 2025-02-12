from typing import Optional

import numpy as np
import numpy.typing as npt


def softmax(x: npt.NDArray[np.float32], axis=1) -> npt.NDArray[np.float32]:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return (1 / (1 + np.exp(-x))).astype(np.float32)


def gaussian_weights(
    size: int, sigma: Optional[float] = None
) -> npt.NDArray[np.float32]:
    """
    Generates a 1D Gaussian weight array.

    Args:
        size (int): Size of the array.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: 1D array of Gaussian weights.
    """
    if sigma is None:
        sigma = size / 4
    vals = np.arange(size) - size // 2
    x, y = np.meshgrid(vals, vals)
    weights = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    return (weights / np.max(weights)).astype(np.float32)


def _get_pad(size: int, tile_size: int, stride: int) -> int:
    if size < tile_size:
        return tile_size - size
    remainder = (size - tile_size) % stride
    if remainder == 0:
        return 0
    return stride - remainder


def get_padding(size: int, tile_size: int, stride: int) -> tuple[int, int]:
    pad = _get_pad(size, tile_size, stride)
    if pad == 0:
        return 0, 0
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after
