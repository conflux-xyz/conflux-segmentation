from typing import Literal

import numpy as np
import numpy.typing as npt

from conflux_segmentation.patch_segmenter import BinaryPatchSegmenterBase
from conflux_segmentation.utils import gaussian_weights, get_padding


class BinarySegmenter:
    def __init__(
        self,
        patch_segmenter: BinaryPatchSegmenterBase,
        patch_size: int = 512,
        overlap: float = 0.125,
        blend_mode: Literal["gaussian", "flat"] = "gaussian",
        pad_value: int = 255,
        batch_size: int = 1,
        threshold: float = 0.5,
    ) -> None:
        self.patch_segmenter = patch_segmenter
        if blend_mode == "gaussian":
            self.blend_weights = gaussian_weights(patch_size)
        else:
            self.blend_weights = np.ones((patch_size, patch_size), dtype=np.float32)
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.stride = round(patch_size * (1 - overlap))
        self.batch_size = batch_size
        self.threshold = threshold

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
        assert image.ndim == 3, "Input image must have 3 dimensions (H x W x C)"
        H, W, _C = image.shape
        pad_y = get_padding(H, self.patch_size, self.stride)
        pad_x = get_padding(W, self.patch_size, self.stride)
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

        # Generate list of image patch coordinates. Add 1 to the endpoint to make it inclusive.
        patch_coords = [
            (y, x)
            for y in range(0, H - self.patch_size + 1, self.stride)
            for x in range(0, W - self.patch_size + 1, self.stride)
        ]

        for patch_coords_batch in [
            patch_coords[i : i + self.batch_size]
            for i in range(0, len(patch_coords), self.batch_size)
        ]:
            # Extract patches from the image (N x H x W x C)
            patches = np.stack(
                [
                    image[y : y + self.patch_size, x : x + self.patch_size]
                    for y, x in patch_coords_batch
                ]
            )
            # Move channel dimension (N x C x H x W)
            patches = np.transpose(patches, (0, 3, 1, 2))
            outputs = self.patch_segmenter(patches)
            for (y, x), output in zip(patch_coords_batch, outputs):
                output_probs[y : y + self.patch_size, x : x + self.patch_size] += (
                    output * self.blend_weights
                )
                output_weights[y : y + self.patch_size, x : x + self.patch_size] += (
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
