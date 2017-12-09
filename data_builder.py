from constants import *
import numpy as np
from os.path import join
import h5py

class data_builder(object):

  def __init__(self):
    pass

  def load_from_save(self):
    with h5py.File(TRAINING_IMAGES_FILENAME, 'r') as hf:
      self._images = hf['name-of-dataset'][:]
    with h5py.File(TRAINING_LABELS_FILENAME, 'r') as hf:
      self._labels = hf['name-of-dataset'][:]
    with h5py.File(VALIDATION_IMAGES_FILENAME, 'r') as hf:
      self._images_validation = hf['name-of-dataset'][:]
    with h5py.File(VALIDATION_LABELS_FILENAME, 'r') as hf:
      self._labels_validation = hf['name-of-dataset'][:]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_validation(self):
    return self._images_validation

  @property
  def labels_validation(self):
    return self._labels_validation