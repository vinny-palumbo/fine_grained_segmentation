import os
import pytest
import numpy as np
import imageio
from tempfile import mkdtemp
import subprocess
import time
import requests

from  fine_grained_segmentation import utils


TEST_IMAGE_FILENAME = "test_image.jpg"
RESULT_FILENAME = "result.png"
ONNX_WEIGHTS_URL = r"https://github.com/vinny-palumbo/fine_grained_segmentation/releases/download/v0.1-alpha/mrcnn.onnx"
ONNX_WEIGHTS_FILENAME = "mrcnn.onnx"


def test_onnx_file_url_exists():

    request = requests.get(ONNX_WEIGHTS_URL)
    assert request.status_code == 200 # Standard response for successful HTTP requests


def test_download_onnx_file():
    
    temp_dir = mkdtemp()
    
    # download onnx weights in library folder
    onnx_file = os.path.join(temp_dir, ONNX_WEIGHTS_FILENAME)
    utils.download_file(ONNX_WEIGHTS_URL, onnx_file)
    
    # get files in temp folder
    temp_filenames = os.listdir(temp_dir)
    
    assert ONNX_WEIGHTS_FILENAME in temp_filenames


def test_segmentation_on_test_image_produces_result_image():
    
    # get test image file
    test_folder = os.path.dirname(os.path.realpath(__file__))
    test_image_file = os.path.join(test_folder, TEST_IMAGE_FILENAME)
    
    # create temporary directory and move to it
    temp_dir = mkdtemp()
    os.chdir(temp_dir)
    
    # run segmentation on test image
    subprocess.call(['fashion-segmentator', '--image', test_image_file])
    
    # assert that result.png was generated in the temporary directory
    temp_filenames = os.listdir(temp_dir)
    
    assert RESULT_FILENAME in temp_filenames