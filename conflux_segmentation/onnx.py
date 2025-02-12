from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]

from .tile_segmenter import TileSegmenterBase
from .utils import sigmoid, softmax
from .types import ActivationType, TilesTransformType
from .defaults import DEFAULT_ACTIVATION, DEFAULT_TILES_TRANSFORM


class OnnxTileSegmenter(TileSegmenterBase):
    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        transform: TilesTransformType = DEFAULT_TILES_TRANSFORM,
        activation: ActivationType = DEFAULT_ACTIVATION,
    ) -> None:
        """
        Wraps a ONNX model for segmentation.
        We assume that given an input of shape (N, C, H, W), the model returns an output of shape (N, K, H, W).
        If the model outputs logits, then you must set `activation="sigmoid"` for binary or multiclass segmentation or
        `activation="softmax"` for multiclass.
        If `activation=None`, we assume the model has already normalized the outputs.
        """
        super().__init__(transform)
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
