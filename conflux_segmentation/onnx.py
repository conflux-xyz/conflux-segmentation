from typing import cast, Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from conflux_segmentation.tile_segmenter import BinaryTileSegmenterBase
from conflux_segmentation.utils import sigmoid


class OnnxBinaryTileSegmenter(BinaryTileSegmenterBase):
    def __init__(
        self,
        session: ort.InferenceSession,
        activation: Literal["sigmoid"] | None = "sigmoid",
    ) -> None:
        self.session = session
        self.activation = activation
        assert len(self.session.get_inputs()) == 1, "Model must have exactly 1 input"
        self.input_name = self.session.get_inputs()[0].name
        assert len(self.session.get_outputs()) >= 1, "Model must have at least 1 output"

    def segment(self, tiles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        ort_inputs = {self.input_name: tiles}
        output = cast(
            npt.NDArray[np.float32],
            self.session.run(output_names=None, input_feed=ort_inputs)[0],
        )
        if self.activation == "sigmoid":
            output = sigmoid(output)
        return output
