import os
import pytest
import numpy as np
import imageio
from tempfile import mkdtemp
import subprocess
import time


TEST_IMAGE_FILENAME = "test_image.jpg"
    
    
def test_running_segmentator_on_test_image():
    
    # get test image file
    pwd_path = os.path.dirname(os.path.realpath(__file__))
    test_image_file = os.path.join(pwd_path, TEST_IMAGE_FILENAME)
    
    # run segmentation on test image
    subprocess.call(['fashion-segmentator', '--image', test_image_file])
    
    # assert that result.png was generated in the parent directory
    parent_path = os.path.abspath(os.path.join(pwd_path, os.pardir))
    filenames = os.listdir(parent_path)
    
    assert "result.png" in filenames