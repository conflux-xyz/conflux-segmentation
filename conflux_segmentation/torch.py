from typing import cast, Literal, Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from conflux_segmentation.patch_segmenter import BinaryPatchSegmenterBase

if TYPE_CHECKING:
    from conflux_segmentation.binary_segmenter import BinarySegmenter


class TorchBinaryPatchSegmenter(BinaryPatchSegmenterBase):
    def __init__(
        self,
        model: torch.nn.Module,
        activation: Optional[Literal["sigmoid"]] = "sigmoid",
    ) -> None:
        self.model = model.eval()
        self.activation = activation

    def segment(self, patches: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x = torch.from_numpy(patches)
        with torch.inference_mode():
            output = self.model(x)
            if self.activation == "sigmoid":
                output = output.sigmoid()
            return cast(npt.NDArray[np.float32], output.cpu().numpy())


def get_binary_segmenter(
    model: torch.nn.Module,
    activation: Literal["sigmoid"] = "sigmoid",
    *,
    patch_size: int = 512,
    overlap: float = 0.125,
    blend_mode: Literal["gaussian", "flat"] = "gaussian",
    pad_value: int = 255,
    batch_size: int = 1,
    threshold: float = 0.5,
) -> "BinarySegmenter":
    from conflux_segmentation.binary_segmenter import BinarySegmenter

    patch_segmenter = TorchBinaryPatchSegmenter(model, activation)
    return BinarySegmenter(
        patch_segmenter,
        patch_size,
        overlap,
        blend_mode,
        pad_value,
        batch_size,
        threshold,
    )
