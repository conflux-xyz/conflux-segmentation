from typing import cast

import numpy as np
import numpy.typing as npt
import torch

from .tile_segmenter import TileSegmenterBase
from .utils import ActivationType


class TorchTileSegmenter(TileSegmenterBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        activation: ActivationType = None,
    ) -> None:
        self.model = model.eval()
        self.activation = activation

    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x = torch.from_numpy(tiles)
        with torch.inference_mode():
            output = self.model(x)
            if self.activation == "sigmoid":
                output = output.sigmoid()
            elif self.activation == "softmax":
                output = output.softmax(dim=1)
            return cast(npt.NDArray[np.float32], output.cpu().numpy())
