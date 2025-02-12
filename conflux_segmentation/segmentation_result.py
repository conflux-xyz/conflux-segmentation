from typing import Sequence

import numpy as np
import numpy.typing as npt

from .defaults import DEFAULT_THRESHOLD


class SegmentationResult:
    def __init__(
        self,
        probabilities: npt.NDArray[np.float32],
    ) -> None:
        # Shape: [H, W, C]
        self.probabilities = probabilities

    def to_binary(self) -> "BinarySegmentationResult":
        assert self.probabilities.shape[2] == 1, (
            "Model must have exactly 1 output channel"
        )
        return BinarySegmentationResult(self.probabilities[:, :, 0])

    def to_multiclass(self) -> "MulticlassSegmentationResult":
        assert self.probabilities.shape[2] > 1, (
            "Model must have more than 1 output channel"
        )
        return MulticlassSegmentationResult(self.probabilities)

    def to_multilabel(self) -> "MultilabelSegmentationResult":
        assert self.probabilities.shape[2] > 1, (
            "Model must have more than 1 output channel"
        )
        return MultilabelSegmentationResult(self.probabilities)


class BinarySegmentationResult:
    def __init__(
        self,
        probabilities: npt.NDArray[np.float32],
    ) -> None:
        self.probabilities = probabilities

    def get_mask(self, threshold: float = DEFAULT_THRESHOLD) -> npt.NDArray[np.bool_]:
        """
        Returns a H x W boolean mask `out` where `out[h,w]` indicates whether the pixel at (h, w)
        is in the positive class.
        """
        return self.probabilities > threshold

    def get_mask_proba(self, copy=True) -> npt.NDArray[np.float32]:
        """
        Returns a H x W array `out` where `out[h,w]` indicates the "probability"
        that the pixel at (h, w) is in the positive class.
        """
        if copy:
            return self.probabilities.copy()
        else:
            return self.probabilities


class MulticlassSegmentationResult:
    def __init__(
        self,
        probabilities: npt.NDArray[np.float32],  # Shape: [H, W, C]
    ) -> None:
        self.probabilities = probabilities

    def get_mask(self) -> npt.NDArray[np.uint]:
        """
        Returns a H x W array `out` where `out[h,w]` indicates the predicted class
        for the pixel at (h, w).
        """
        return np.argmax(self.probabilities, axis=2).astype(np.uint)

    def get_mask_proba(self, copy=True) -> npt.NDArray[np.float32]:
        """
        Returns a H x W x C array `out` where `out[h,w,c]` indicates the "probability"
        that the pixel at (h, w) is in class c.
        """
        if copy:
            return self.probabilities.copy()
        else:
            return self.probabilities


class MultilabelSegmentationResult:
    def __init__(
        self,
        probabilities: npt.NDArray[np.float32],  # Shape: [H, W, C]
    ) -> None:
        self.probabilities = probabilities

    def get_mask(
        self, threshold: float | Sequence[float] = DEFAULT_THRESHOLD
    ) -> npt.NDArray[np.bool_]:
        """
        Returns a H x W x C array `out` where `out[h,w,c]` whether the pixel at (h, w)
        belongs to label c.
        """
        if isinstance(threshold, float):
            return self.probabilities > threshold
        else:
            assert len(threshold) == self.probabilities.shape[2], (
                "Threshold must have the same length as number of classes"
            )
            return self.probabilities > np.array(threshold)[None, None, :]

    def get_mask_proba(self, copy=True) -> npt.NDArray[np.float32]:
        """
        Returns a H x W x C array `out` where `out[h,w,c]` indicates the "probability"
        that the pixel at (h, w) has label c
        """
        if copy:
            return self.probabilities.copy()
        else:
            return self.probabilities
