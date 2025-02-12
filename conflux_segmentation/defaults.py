import numpy as np
import numpy.typing as npt

from .types import ActivationType, BlendModeType

DEFAULT_NUM_CLASSES: int = 1
DEFAULT_TILE_SIZE: int = 512
DEFAULT_OVERLAP: float = 0.125
DEFAULT_BLEND_MODE: BlendModeType = "gaussian"
DEFAULT_PAD_VALUE: int = 255
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_ACTIVATION: ActivationType = None
DEFAULT_THRESHOLD: float = 0.5


def DEFAULT_TILES_TRANSFORM(tiles: npt.NDArray[np.uint8]):
    return (tiles / 255).astype(np.float32)
