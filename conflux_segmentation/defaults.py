from .types import ActivationType, BlendModeType
from .transforms import ToFloat32

DEFAULT_NUM_CLASSES: int = 1
DEFAULT_TILE_SIZE: int = 512
DEFAULT_OVERLAP: float = 0.125
DEFAULT_BLEND_MODE: BlendModeType = "gaussian"
DEFAULT_PAD_VALUE: int = 255
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_ACTIVATION: ActivationType = None
DEFAULT_THRESHOLD: float = 0.5

DEFAULT_TILES_TRANSFORM = ToFloat32()
