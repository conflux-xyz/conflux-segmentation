from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from .tile_segmenter import TileSegmenterBase
from .utils import ActivationType, sigmoid, softmax


class OnnxBinaryTileSegmenter(TileSegmenterBase):
    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        activation: ActivationType = None,
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
        elif self.activation == "softmax":
            output = softmax(output, axis=1)
        return output
