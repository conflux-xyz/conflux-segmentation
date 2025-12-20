from typing import Sequence
import numpy as np
import numpy.typing as npt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize:
    """
    Normalize tiles using the given mean and std.
    Similar to torchvision.transforms.Normalize (https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html).
    and albumentations.Normalize (https://explore.albumentations.ai/transform/Normalize).
    """

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        max_pixel_value: float = 255.0,
    ):
        dims = len(mean)
        assert dims == len(std), "Mean and std must have the same length"
        self.dims = dims
        self.mean = (np.array(mean).reshape(1, -1, 1, 1) * max_pixel_value).astype(
            np.float32
        )
        self.std = (np.array(std).reshape(1, -1, 1, 1) * max_pixel_value).astype(
            np.float32
        )

    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        assert tiles.ndim == 4, "Input tiles must have 4 dimensions (N, C, H, W)"
        assert tiles.shape[1] == self.dims, (
            f"Input tiles must have {self.dims} channels"
        )
        # Transform tiles (N, C, H, W) from uint8 [0, 255] to float32 normalized
        return (tiles.astype(np.float32) - self.mean) / self.std


class ToFloat32:
    """
    Convert tiles from uint8 [0, max_pixel_value] (generally [0, 255]) to float32 [0.0, 1.0].
    """

    def __init__(self, max_pixel_value: float = 255.0):
        self.max_pixel_value = max_pixel_value

    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return tiles.astype(np.float32) / self.max_pixel_value
