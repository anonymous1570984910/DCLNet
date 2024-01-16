<p align="center">

  <h1>DCLNet: Data Closed-Loop Network Model</h1>

## 📢 Location of key codes

In `./auto_anno.py`, we provide the implementations of “Grad-CAM & SAM Automatic Annotation” method (Ours).

In `./dataset_gen.py`, we provide the implementations of generating curriculum learning datasets for the subsequent data closed-loop process.

We provide all the configuration files in the paper in configs/.

## 🏃‍♂️ Getting Started

Download the pretrained base models for [YOLO V5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) and [Segment Anything](https://github.com/facebookresearch/segment-anything).

Place your data as follows:
```bash
DCLNet
|----datasets
  |----Laryngoscope8_cls (dataset name)
    |----0 (cls_name)
      |----00000.png
      |----00001.png
      |----...
    |----1
      |----10000.png
      |----10001.png
      |----...
    |----2
      |----20000.png
      |----20001.png
      |----...
    |----...
  |----Laryngoscope8_raw(dataset name)
    |----00000.png
    |----00001.png
    |----10000.png
    |----10001.png
    |----20000.png
    |----20001.png
    |----...
```

## ⚒️ Installation
prerequisites: `python>=3.8`, `pytorch>=1.8`, and `torchvision>=0.8`.

Install with `pip`:
```bash
git clone https://github.com/anonymous1570984910/DCLNet  # clone
cd DCLNet
pip install -r requirements.txt # install
```

## 🎨 Preprocess
Train a resnet18 model as a starting point:

```bash
python ./models/resnet18/train.py --img_path path/to/your/dataset_cls --save_dir path/to/save/weights
```
Generate activation maps:

```bash
python ./models/resnet18/gradcam.py --img_path path/to/your/dataset_cls --weights_path path/to/your/weights --save_dir path/to/save/maps
```

## 💃 Data Closed-Loop 

Generate masks:

```bash
python ./models/segment_anything/scripts/amg.py --input path/to/your/dataset_raw --output path/to/save/masks
```

In SAM, --points-per-side, --pred-iou-thresh and --stability-score-thresh parameters can be adjusted to achieve better results.

Automatic annotation:

```bash
python ./auto_anno.py --img_path path/to/your/dataset_raw --sam_path path/to/your/sam/results --cam_path path/to/your/cam/results --save_txt path/to/save/annotations
```
Train YOLOV5_single:
```bash
python ./models/yolov5/main/train.py --weights path/to/your/weights --cfg path/to/your/cfg/file --data path/to/your/data_cfg/file
```
Detection:

```bash
python ./models/yolov5/main/detect.py --weights path/to/saved/weights --source path/to/your/dataset
```

Generate Curriculum Learning Dataset:

```bash
python ./dataset_gen.py --label_path path/to/saved/labels --ratio_thre data/volume/ratio --dataset_path path/to/your/dataset_raw --save_dir path/to/save/dataset
```

Train YOLOV5_mul:

```bash
python ./models/yolov5/main/train.py --weights path/to/your/weights --cfg path/to/your/cfg/file --data path/to/your/data_cfg/file
```

Generate activation maps:

```bash
python ./generate.py --model-path path/to/saved/weights --img-path path/to/your/dataset_raw --output-dir path/to/save/maps
```

Multiple iterations of the Data Closed-Loop can be employed to achieve better results.

## 🙏 Acknowledgements
Our model is built based on YOLO and SAM, and the specific details and introduction of the code can be found in their GitHub repository: [YOLO V5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) and [Segment Anything](https://github.com/facebookresearch/segment-anything).
