# Conflux Segmentation

A Python library for tile-based inference for segmentation of large images.

Assuming you have a segmentation model that operates on tiles (e.g. 512 x 512), this library provides the plumbing to apply that model on a large image and handles the padding, striding, and blending to apply the model across the entire image.

## Usage

### Binary Segmentation

First, construct the `BinarySegmenter`:

For [PyTorch](https://pytorch.org/) (e.g. with [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/)):

```python
# $ pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
from conflux_segmentation import BinarySegmenter

net = smp.Unet(encoder_name="tu-mobilenetv3_small_100", encoder_weights=None, activation=None)
net.load_state_dict(torch.load("/path/to/weights", weights_only=True))
net.eval()
segmenter = BinarySegmenter.from_pytorch_module(net)
```

Or, for [ONNX Runtime](https://onnxruntime.ai/):

```python
import onnxruntime as ort
from conflux_segmentation import BinarySegmenter

session = ort.InferenceSession("/path/to/model.onnx")
segmenter = BinarySegmenter.from_onnxruntime_session(session)
```

Then, to segment a large image:

```python
# $ pip install opencv-python-headless
import cv2

# H x W x 3 image array of np.uint8
image = cv2.cvtColor(cv2.imread("/path/to/large/image"), cv2.COLOR_BGR2RGB)

result = segmenter(image)
# H x W boolean array
mask = result.get_mask()
assert mask.shape == image.shape[:2]
assert (mask == True).sum() + (mask == False).sum() == mask.size
```