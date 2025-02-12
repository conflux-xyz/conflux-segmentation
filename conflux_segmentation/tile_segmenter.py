from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class TileSegmenterBase(ABC):
    @abstractmethod
    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    def transform(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return (tiles / 255).astype(np.float32)

    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """
        Assumes tiles is N x C x H x W
        Returns an array of shape N x K x H x W
        """
        assert tiles.ndim == 4, (
            "Input image tiles must have 4 dimensions (N x C x H x W)"
        )
        N, H, W, _C = tiles.shape
        # assert C == 3, "Input image tiles must have 3 channels (RGB)"
        # Move channel dimension (N x H x W x C => N x C x H x W)
        tiles = np.transpose(tiles, (0, 3, 1, 2))
        # Output shape is N x K x H x W
        output = self.segment(self.transform(tiles))
        assert output.ndim == 4, "Output mask must have 4 dimensions (N x K x H x W)"
        assert output.shape[0] == N, (
            "Output mask must have the same number of samples as the input image"
        )
        assert output.shape[2:] == (H, W), (
            "Output mask must have the same shape as the input image"
        )
        # Move class dimension (N x K x H x W => N x H x W x K)
        output = np.transpose(output, (0, 2, 3, 1))
        return output
