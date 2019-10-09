# Fine-Grained Segmentation

This is a project for fine-grained segmentation on clothing items in images, implemented in Python 3, Keras and TensorFlow. A deep learning model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on [Matterport Mask R-CNN](https://github.com/matterport/Mask_RCNN)

## Requirements

Python 3.5, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

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
3. Run setup in the Mask_RCNN directory
    ```bash
    cd fine_grained_segmentation/Mask_RCNN
	python3 setup.py install
    ``` 
4. (Optional) Download pre-trained COCO weights (mask_rcnn_coco.h5) from the Matterport Mask_RCNN [releases page](https://github.com/matterport/Mask_RCNN/releases) and place it in the fine_grained_segmentation/Mask_RCNN folder

## Getting Started

* [detect.py](fine_grained_segmentation/model/detect.py) detects and segments fashion items in images, given a folder of images and pre-trained weights

* ([model.py](fine_grained_segmentation/Mask_RCNN/mrcnn/model.py), [utils.py](fine_grained_segmentation/Mask_RCNN/mrcnn/utils.py), [config.py](fine_grained_segmentation/Mask_RCNN/mrcnn/config.py)): These files contain the main Mask RCNN implementation. 



