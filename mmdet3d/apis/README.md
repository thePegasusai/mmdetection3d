# mmdet3d API

### To create and init model

```
from mmdet3d.apis.inference import init_model
from mmdet3d.apis.utils import find_config

config = find_config('centerpoint/centerpoint_03pillar_kitti_lum.py')
model = init_model(config, checkpoint, device='cuda:0')
```

### To perform inference

```
from mmdet3d.apis import init_model, inference_detector

pcd = '/home/mark/KITTI/testing/velodyne/000008.bin'
model = init_model(config, checkpoint, device='cuda:0')
result, data = inference_detector(model, pcd)
```

### To convert model to ONNX format

```
from mmdet3d.apis.convert import convert_to_onnx

convert_to_onnx(config, checkpoint=checkpoint, verbose=True)
```

### To build TensorRT engine

```
from mmdet3d.apis.trt_engine import build_trt_engine, test_trt_engine

build_trt_engine('path to ONNX export (.onnx)')
test_trt_engine('path to TensorRT engine (.trt)')
```

### To run CenterPoint conversion

```commandline
python3 mmdet3d/apis/convert.py
python3 mmdet3d/apis/trt_engine.py
```