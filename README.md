# Fine-Grained Segmentation

This is a project for fine-grained segmentation on clothing items in images, implemented in Python 3 and ONNX. A deep learning model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN)

## Requirements

Python 3.5, ONNX runtime, and other common packages listed in `requirements.txt`.

## Installation

1. Clone this repository
2. Run setup to install the library
   ```bash
   python3 setup.py install
   ```
   If it failed to install the dependencies, run
   ```bash
   pip3 install -r requirements.txt
   ```

## Getting Started

* [detect.py](fine_grained_segmentation/model/detect.py) detects and segments fashion items in an image




