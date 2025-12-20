from typing import Sequence
import numpy as np
import numpy.typing as npt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize:
    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        max_pixel_value: float = 255.0,
    ):
        dims = len(mean)
        assert dims == len(std), "Mean and std must have the same length"
        self.dims = dims
        self.mean = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
        self.max_pixel_value = max_pixel_value

    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        assert tiles.ndim == 4, "Input tiles must have 4 dimensions (N, C, H, W)"
        assert tiles.shape[1] == self.dims, (
            f"Input tiles must have {self.dims} channels"
        )
        # Transform tiles (N, C, H, W) from uint8 [0, 255] to float32 normalized
        tiles32 = tiles.astype(np.float32) / self.max_pixel_value
        return (tiles32 - self.mean) / self.std


class ToFloat32:
    def __init__(self, max_pixel_value: float = 255.0):
        self.max_pixel_value = max_pixel_value

    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return tiles.astype(np.float32) / self.max_pixel_value
