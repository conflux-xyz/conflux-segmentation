from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class PatchSegmenterBase(ABC):
    @abstractmethod
    def segment(self, patches: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    def transform(self, patches: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        return (patches / 255).astype(np.float32)

    def __call__(self, patches: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        assert patches.ndim == 4, (
            "Input image patches must have 4 dimensions (N x C x H x W)"
        )
        N, C, H, W = patches.shape
        assert C == 3, "Input image patches must have 3 channels (RGB)"
        # x_tensor = torch.from_numpy(x) / 255
        # with torch.inference_mode():
        #     output = cast(npt.NDArray[np.float32], self.model(x_tensor).sigmoid().cpu().numpy())
        output = self.segment(self.transform(patches))
        assert output.ndim == 4, "Output mask must have 4 dimensions (N x K x H x W)"
        assert output.shape[0] == N, (
            "Output mask must have the same number of samples as the input image"
        )
        assert output.shape[2:] == (H, W), (
            "Output mask must have the same shape as the input image"
        )
        return output


class BinaryPatchSegmenterBase(PatchSegmenterBase):
    def __call__(self, patches: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        output = super().__call__(patches)
        assert output.shape[1] == 1, "Binary segmentation must have 1 channel"
        return output.squeeze(1)
