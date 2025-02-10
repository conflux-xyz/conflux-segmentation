from typing import Literal, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from conflux_segmentation.tile_segmenter import BinaryTileSegmenterBase
from conflux_segmentation.utils import gaussian_weights, get_padding


if TYPE_CHECKING:
    import torch
    import onnxruntime as ort  # type: ignore[import-untyped]


class BinarySegmenter:
    def __init__(
        self,
        tile_segmenter: BinaryTileSegmenterBase,
        tile_size: int = 512,
        overlap: float = 0.125,
        blend_mode: Literal["gaussian", "flat"] = "gaussian",
        pad_value: int = 255,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> None:
        self.tile_segmenter = tile_segmenter
        if blend_mode == "gaussian":
            self.blend_weights = gaussian_weights(tile_size)
        else:
            self.blend_weights = np.ones((tile_size, tile_size), dtype=np.float32)
        self.tile_size = tile_size
        self.pad_value = pad_value
        self.stride = round(tile_size * (1 - overlap))
        self.batch_size = batch_size
        self.threshold = threshold

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        assert image.ndim == 3, "Input image must have 3 dimensions (H x W x C)"
        H, W, _C = image.shape
        pad_y = get_padding(H, self.tile_size, self.stride)
        pad_x = get_padding(W, self.tile_size, self.stride)
        # Padding with 255 is important -- that is how it was trained.
        image_padded = np.pad(
            image,
            (pad_y, pad_x, (0, 0)),
            mode="constant",
            constant_values=self.pad_value,
        )
        mask_padded = self._segment(image_padded)
        mask = mask_padded[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
        return mask

    def _segment(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        H, W, _C = image.shape

        # Initialize the output mask
        output_probs = np.zeros((H, W), dtype=np.float32)
        output_weights = np.zeros((H, W), dtype=np.float32)

        # Generate list of image tile coordinates. Add 1 to the endpoint to make it inclusive.
        tile_coords = [
            (y, x)
            for y in range(0, H - self.tile_size + 1, self.stride)
            for x in range(0, W - self.tile_size + 1, self.stride)
        ]

        for tile_coords_batch in [
            tile_coords[i : i + self.batch_size]
            for i in range(0, len(tile_coords), self.batch_size)
        ]:
            # Extract tiles from the image (N x H x W x C)
            tiles = np.stack(
                [
                    image[y : y + self.tile_size, x : x + self.tile_size]
                    for y, x in tile_coords_batch
                ]
            )
            # Move channel dimension (N x C x H x W)
            tiles = np.transpose(tiles, (0, 3, 1, 2))
            outputs = self.tile_segmenter(tiles)
            for (y, x), output in zip(tile_coords_batch, outputs):
                output_probs[y : y + self.tile_size, x : x + self.tile_size] += (
                    output * self.blend_weights
                )
                output_weights[y : y + self.tile_size, x : x + self.tile_size] += (
                    self.blend_weights
                )

        return (
            np.divide(
                output_probs,
                output_weights,
                out=np.zeros_like(output_probs),
                where=output_weights != 0,
            )
            > self.threshold
        )

    @staticmethod
    def from_pytorch_module(
        model: "torch.nn.Module",
        activation: Literal["sigmoid"] = "sigmoid",
        *,
        tile_size: int = 512,
        overlap: float = 0.125,
        blend_mode: Literal["gaussian", "flat"] = "gaussian",
        pad_value: int = 255,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> "BinarySegmenter":
        from .torch import TorchBinaryTileSegmenter

        tile_segmenter = TorchBinaryTileSegmenter(model, activation)
        return BinarySegmenter(
            tile_segmenter,
            tile_size,
            overlap,
            blend_mode,
            pad_value,
            batch_size,
            threshold,
        )

    @staticmethod
    def from_onnxruntime_session(
        session: "ort.InferenceSession",
        activation: Literal["sigmoid"] | None = "sigmoid",
        *,
        tile_size: int = 512,
        overlap: float = 0.125,
        blend_mode: Literal["gaussian", "flat"] = "gaussian",
        pad_value: int = 255,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> "BinarySegmenter":
        from .onnx import OnnxBinaryTileSegmenter

        tile_segmenter = OnnxBinaryTileSegmenter(session, activation)
        return BinarySegmenter(
            tile_segmenter,
            tile_size,
            overlap,
            blend_mode,
            pad_value,
            batch_size,
            threshold,
        )
