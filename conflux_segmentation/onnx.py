from typing import cast, Literal, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from conflux_segmentation.patch_segmenter import BinaryPatchSegmenterBase
from conflux_segmentation.utils import sigmoid

if TYPE_CHECKING:
    from conflux_segmentation.binary_segmenter import BinarySegmenter


class OnnxBinaryPatchSegmenter(BinaryPatchSegmenterBase):
    def __init__(
        self,
        session: ort.InferenceSession,
        activation: Literal["sigmoid"] | None = "sigmoid",
    ) -> None:
        self.session = session
        self.activation = activation

    def segment(self, patches: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        ort_inputs = {"input": patches}
        output = cast(
            npt.NDArray[np.float32],
            self.session.run(output_names=None, input_feed=ort_inputs)[0],
        )
        if self.activation == "sigmoid":
            output = sigmoid(output)
        return output


def get_binary_segmenter(
    session: ort.InferenceSession,
    activation: Literal["sigmoid"] | None = "sigmoid",
    *,
    patch_size: int = 512,
    overlap: float = 0.125,
    blend_mode: Literal["gaussian", "flat"] = "gaussian",
    pad_value: int = 255,
    batch_size: int = 1,
    threshold: float = 0.5,
) -> "BinarySegmenter":
    from conflux_segmentation.binary_segmenter import BinarySegmenter

    patch_segmenter = OnnxBinaryPatchSegmenter(session, activation)
    return BinarySegmenter(
        patch_segmenter,
        patch_size,
        overlap,
        blend_mode,
        pad_value,
        batch_size,
        threshold,
    )
