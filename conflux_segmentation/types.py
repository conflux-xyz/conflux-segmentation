from typing import Callable, Literal, Optional

import numpy as np
import numpy.typing as npt

ActivationType = Optional[Literal["sigmoid", "softmax"]]
BlendModeType = Literal["gaussian", "flat"]
TilesTransformType = Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]]
