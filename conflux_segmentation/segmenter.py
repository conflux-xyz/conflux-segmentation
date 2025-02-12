from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt

from .tile_segmenter import TileSegmenterBase
from .segmentation_result import SegmentationResult
from .utils import gaussian_weights, get_padding
from .types import ActivationType, BlendModeType
from .defaults import (
    DEFAULT_ACTIVATION,
    DEFAULT_NUM_CLASSES,
    DEFAULT_TILE_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_BLEND_MODE,
    DEFAULT_PAD_VALUE,
    DEFAULT_BATCH_SIZE,
)

if TYPE_CHECKING:
    import torch
    import onnxruntime as ort  # type: ignore[import-untyped]


class Segmenter:
    def __init__(
        self,
        tile_segmenter: TileSegmenterBase,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        blend_mode: BlendModeType = DEFAULT_BLEND_MODE,
        pad_value: int = DEFAULT_PAD_VALUE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        assert num_classes > 0, "Number of classes must be greater than 0"
        self.num_classes = num_classes
        self.tile_segmenter = tile_segmenter
        if blend_mode == "gaussian":
            # Expand dims to [H, W, 1] to broadcast across classes
            self.blend_weights = gaussian_weights(tile_size)[..., None]
        else:
            self.blend_weights = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        self.tile_size = tile_size
        self.stride = round(tile_size * (1 - overlap))
        self.pad_value = pad_value
        self.batch_size = batch_size

    def __call__(self, image: npt.NDArray[np.uint8]) -> SegmentationResult:
        return SegmentationResult(self._segment(image))

    def _segment(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        assert image.ndim == 3, "Input image must have 3 dimensions (H x W x C)"
        H, W, _C = image.shape
        pad_y = get_padding(H, self.tile_size, self.stride)
        pad_x = get_padding(W, self.tile_size, self.stride)
        image_padded = np.pad(
            image,
            (pad_y, pad_x, (0, 0)),
            mode="constant",
            constant_values=self.pad_value,
        )
        probs_padded = self._segment_padded(image_padded)
        probs = probs_padded[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
        return probs

    def _segment_padded(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        H, W, _C = image.shape

        # Initialize outputs with class dimension
        output_probs = np.zeros((H, W, self.num_classes), dtype=np.float32)
        output_weights = np.zeros((H, W, self.num_classes), dtype=np.float32)

        # Generate tile coordinates
        tile_coords = [
            (y, x)
            for y in range(0, H - self.tile_size + 1, self.stride)
            for x in range(0, W - self.tile_size + 1, self.stride)
        ]

        for tile_coords_batch in [
            tile_coords[i : i + self.batch_size]
            for i in range(0, len(tile_coords), self.batch_size)
        ]:
            # Extract tiles from image (N x H x W x C)
            tiles = np.stack(
                [
                    image[y : y + self.tile_size, x : x + self.tile_size]
                    for y, x in tile_coords_batch
                ]
            )
            # Get predictions (N x H x W x num_classes)
            outputs = self.tile_segmenter(tiles)

            for (y, x), output in zip(tile_coords_batch, outputs):
                output_probs[y : y + self.tile_size, x : x + self.tile_size] += (
                    output
                    * self.blend_weights  # blend_weights broadcasts to [H,W,num_classes]
                )
                output_weights[y : y + self.tile_size, x : x + self.tile_size] += (
                    self.blend_weights
                )

        # Blend probabilities
        probs = np.divide(
            output_probs,
            output_weights,
            out=np.zeros_like(output_probs, dtype=np.float32),
            where=output_weights != 0,
        ).astype(np.float32)
        return probs

    @staticmethod
    def from_torch(
        model: "torch.nn.Module",
        *,
        activation: ActivationType = DEFAULT_ACTIVATION,
        device: Optional["torch.device"] = None,
        num_classes: int = DEFAULT_NUM_CLASSES,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        blend_mode: BlendModeType = DEFAULT_BLEND_MODE,
        pad_value: int = DEFAULT_PAD_VALUE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "Segmenter":
        from .torch import TorchTileSegmenter

        tile_segmenter = TorchTileSegmenter(model, activation=activation, device=device)
        return Segmenter(
            tile_segmenter,
            num_classes=num_classes,
            tile_size=tile_size,
            overlap=overlap,
            blend_mode=blend_mode,
            pad_value=pad_value,
            batch_size=batch_size,
        )

    @staticmethod
    def from_onnx(
        session: "ort.InferenceSession",
        *,
        activation: ActivationType = DEFAULT_ACTIVATION,
        num_classes: int = DEFAULT_NUM_CLASSES,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: float = DEFAULT_OVERLAP,
        blend_mode: BlendModeType = DEFAULT_BLEND_MODE,
        pad_value: int = DEFAULT_PAD_VALUE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> "Segmenter":
        from .onnx import OnnxTileSegmenter

        tile_segmenter = OnnxTileSegmenter(session, activation=activation)
        return Segmenter(
            tile_segmenter,
            num_classes=num_classes,
            tile_size=tile_size,
            overlap=overlap,
            blend_mode=blend_mode,
            pad_value=pad_value,
            batch_size=batch_size,
        )
