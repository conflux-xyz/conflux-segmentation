# Conflux Segmentation

A Python library for tile-based inference for segmentation of large images.

Assuming you have a segmentation model that operates on tiles (e.g. 512 x 512), this library provides the plumbing to apply that model on a large image -- handling the padding, striding, and blending required.

## Usage

The main `Segmenter` class assumes that the underlying tile-based segmenter outputs a multidimensional array of shape N x K x H x W where H and W are the height and width of a tile (e.g. 512), N is the batch size, and K is the output dimension (e.g. 1 for binary and > 1 for multiclass or multilabel).

Below we show an example of binary segmentation, although multiclass and multilabel are also supported. In this case, we assume the tile model outputs logits, so we specify `"sigmoid"` for the activation.

First, construct the `Segmenter`:

For [PyTorch](https://pytorch.org/) (e.g. with [Segmentation Models PyTorch](https://smp.readthedocs.io/en/latest/)):

```python
# $ pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
import torch
from conflux_segmentation import Segmenter

net = smp.Unet(encoder_name="tu-mobilenetv3_small_100", encoder_weights=None, activation=None)
net.load_state_dict(torch.load("/path/to/weights", weights_only=True))
net.eval()
segmenter = Segmenter.from_torch(net, activation="sigmoid")
# Alternatively, if your model already has a Sigmoid layer at the end:
# import torch.nn as nn
# sigmoid_net = nn.Sequential(net, nn.Sigmoid()).eval()
# segmenter = Segmenter.from_torch(net)
```

Or, for [ONNX Runtime](https://onnxruntime.ai/):

```python
import onnxruntime as ort
from conflux_segmentation import Segmenter

session = ort.InferenceSession("/path/to/model.onnx")
segmenter = Segmenter.from_onnx(session, activation="sigmoid")
```

Then, to segment a large image:

```python
# $ pip install opencv-python-headless
import cv2

# H x W x 3 image array of np.uint8
image = cv2.cvtColor(cv2.imread("/path/to/large/image"), cv2.COLOR_BGR2RGB)

result = segmenter(image).to_binary()
# H x W boolean array
mask = result.get_mask()
assert mask.shape == image.shape[:2]
assert (mask == True).sum() + (mask == False).sum() == mask.size
```