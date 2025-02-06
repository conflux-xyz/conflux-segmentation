# Conflux Segmentation

A Python library for patch-based segmentation of large images.

## Usage

### Binary Segmentation

First, construct the `BinarySegmenter`:

For PyTorch (with Segmentation Models PyTorch):

```python
import segmentation_models_pytorch as smp
from conflux_segmentation.torch import get_binary_segmenter

net = smp.Unet(encoder_name="tu-mobilenetv3_small_100", encoder_weights=None, activation=None)
net.load_state_dict(torch.load("/path/to/weights", weights_only=True))
net.eval()
segmenter = get_binary_segmenter(net)
```

Or, for ONNX:

```python
import onnxruntime as ort
from conflux_segmentation.onnx import get_binary_segmenter

session = ort.InferenceSession("/path/to/model.onnx")
segmenter = get_binary_segmenter(session)
```

Then, to segment a large image in patches:

```python
import cv2

# H x W x 3 image array of np.uint8
image = cv2.cvtColor(cv2.imread("/path/to/large/image"), cv2.COLOR_BGR2RGB)

# H x W boolean array
mask = segmenter(image)
assert mask.shape == image.shape[:2]
assert (mask == True).sum() + (mask == False).sum() == mask.size
```