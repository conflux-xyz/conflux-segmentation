from typing import cast, Optional

import numpy as np
import numpy.typing as npt
import torch

from .tile_segmenter import TileSegmenterBase
from .types import ActivationType, TilesTransformType
from .defaults import DEFAULT_ACTIVATION, DEFAULT_TILES_TRANSFORM


class TorchTileSegmenter(TileSegmenterBase):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        transform: TilesTransformType = DEFAULT_TILES_TRANSFORM,
        activation: ActivationType = DEFAULT_ACTIVATION,
        device: Optional[torch.device],
    ) -> None:
        """
        Wraps a PyTorch model for segmentation.
        We assume that given an input of shape (N, C, H, W), the model returns an output of shape (N, K, H, W).
        If the model outputs logits, then you must set `activation="sigmoid"` for binary or multiclass segmentation or
        `activation="softmax"` for multiclass.
        If `activation=None`, we assume the model has already normalized the outputs.
        """
        super().__init__(transform)
        self.model = model.eval()
        self.activation = activation
        self.device = device if device else torch.get_default_device()
        self.model.to(self.device)

    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x = torch.from_numpy(tiles).to(self.device)
        with torch.inference_mode():
            output = self.model(x)
            if self.activation == "sigmoid":
                output = output.sigmoid()
            elif self.activation == "softmax":
                output = output.softmax(dim=1)
            return cast(npt.NDArray[np.float32], output.cpu().numpy())
