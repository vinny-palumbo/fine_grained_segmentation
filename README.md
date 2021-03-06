# fine-grained-segmentation library [![CircleCI](https://circleci.com/gh/vinny-palumbo/fine_grained_segmentation.svg?style=svg)](https://circleci.com/gh/vinny-palumbo/fine_grained_segmentation)

Python library for segmenting clothing items in images, implemented in Python 3 and ONNX. A deep learning model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN)

The library is also available on the [Python Package Index](https://pypi.org/project/fine-grained-segmentation/)

A demo web app is up at https://fine-grained-segmentation.vinnypalumbo.com

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
   
## Usage

Here is how to use the library from the command line:
```bash
fashion-segmentator --image=<path/to/image/file>
```
This will generate a ```result.png``` file in the current directory
