from typing import cast, Literal, Optional

import numpy as np
import numpy.typing as npt
import torch

from conflux_segmentation.tile_segmenter import BinaryTileSegmenterBase


class TorchBinaryTileSegmenter(BinaryTileSegmenterBase):
    def __init__(
        self,
        model: torch.nn.Module,
        activation: Optional[Literal["sigmoid"]] = "sigmoid",
    ) -> None:
        self.model = model.eval()
        self.activation = activation

    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x = torch.from_numpy(tiles)
        with torch.inference_mode():
            output = self.model(x)
            if self.activation == "sigmoid":
                output = output.sigmoid()
            return cast(npt.NDArray[np.float32], output.cpu().numpy())
