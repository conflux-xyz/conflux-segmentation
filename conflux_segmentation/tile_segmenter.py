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
        assert tiles.ndim == 4, (
            "Input image tiles must have 4 dimensions (N x C x H x W)"
        )
        N, C, H, W = tiles.shape
        assert C == 3, "Input image tiles must have 3 channels (RGB)"
        # x_tensor = torch.from_numpy(x) / 255
        # with torch.inference_mode():
        #     output = cast(npt.NDArray[np.float32], self.model(x_tensor).sigmoid().cpu().numpy())
        output = self.segment(self.transform(tiles))
        assert output.ndim == 4, "Output mask must have 4 dimensions (N x K x H x W)"
        assert output.shape[0] == N, (
            "Output mask must have the same number of samples as the input image"
        )
        assert output.shape[2:] == (H, W), (
            "Output mask must have the same shape as the input image"
        )
        return output


class BinaryTileSegmenterBase(TileSegmenterBase):
    def __call__(self, tiles: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        output = super().__call__(tiles)
        assert output.shape[1] == 1, "Binary segmentation must have 1 channel"
        return output.squeeze(1)
