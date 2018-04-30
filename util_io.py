"""
This file contains utility functions for general purposes file reading.
"""
import Queue
import bisect
import collections
import copy
import errno
import hashlib
import os
import random
import threading
import time
import urllib
from operator import mul
from os.path import basename, dirname
from PIL import Image

import numpy as np
import scipy.misc
from typing import Union, List, Set, Optional

def touch_folder(file_path):
  # type: (Union[str,unicode]) -> None
  """Create a folder along with its parent folders recursively if they do not exist."""
  # Taken from https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist .
  if not file_path.endswith('/'):
    file_path = file_path + "/"
  dn = dirname(file_path)
  if dn != '':
    try:
      os.makedirs(dn)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


def get_all_image_paths(directory, do_sort=True, allowed_extensions = {'.jpg', '.png', '.jpeg'}):
    # type: (str) -> List[str]
    """

    :param directory: The parent directory of the images, or a file containing paths to images.
    :return: A sorted list of paths to images in the directory as well as all of its subdirectories.
    """
    if os.path.isdir(directory):
        # allowed_extensions = {'.jpg', '.png', '.jpeg'}
        if not directory.endswith('/'):
            directory = directory + "/"
        content_dirs = []
        for path, subdirs, files in os.walk(directory):
            for name in files:
                full_file_path = os.path.join(path, name)
                _, ext = os.path.splitext(full_file_path)
                ext = ext.lower()
                if ext in allowed_extensions:
                    content_dirs.append(full_file_path)
        if len(content_dirs) == 0:
            raise AssertionError('There is no image in directory %s.' % directory)
    elif os.path.isfile(directory):
        content_dirs = []
        with open(directory, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0:
                    content_dirs.append(line)
        if len(content_dirs) == 0:
            raise AssertionError('There is no image in file %s.' % directory)
    else:
        raise AssertionError('There is no file or directory named %s.' % directory)
    if do_sort:
        content_dirs.sort()
    return content_dirs


def imread(path, shape=None, bw=False, rgba=False, dtype=np.float32):
    # type: (str, tuple, bool, bool) -> np.ndarray
    """

    :param path: path to the image
    :param shape: (Height, width)
    :param bw: Whether the image is black and white.
    :param rgba: Whether the image is in rgba format.
    :return: np array with shape (height, width, num_color(1, 3, or 4))
    """
    assert not (bw and rgba)
    if bw:
        convert_format = 'L'
    elif rgba:
        convert_format = 'RGBA'
    else:
        convert_format = 'RGB'

    if shape is None:
        return np.asarray(Image.open(path).convert(convert_format), dtype)
    else:
        return np.asarray(Image.open(path).convert(convert_format).resize((shape[1], shape[0])), dtype)


def imsave(path, img):
    # type: (str, np.ndarray) -> None
    """
    Automatically clip the image represented in a numpy array to 0~255 and save the image.
    :param path: Path to save the image.
    :param img: Image represented in numpy array with a legal format for scipy.misc.imsave
    :return: None
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, -1)
    scipy.misc.imsave(path, img)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])