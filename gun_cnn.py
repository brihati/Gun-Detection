from __future__ import division, absolute_import

import sys
from os.path import isfile, join
import tflearn
import constants
from data_builder import data_builder
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization


class gun_cnn:
  def __init__(self):
    self.dataset = data_builder()

  def build_network(self):
    print('---------------------Building CNN---------------------')
    img_preProcess=ImagePreprocessing()
    img_preProcess.add_featurewise_zero_center()
    img_preProcess.add_featurewise_stdnorm()

    # Mean: 189.80002318
    # STD: 85.4885473338
    #img_aug=ImageAugmentation()
    #img_aug.add_random_flip_leftright()
    #img_aug.add_random_rotation(max_angle=25)
    #img_aug.add_random_blur(sigma_max=3)

    self.network = input_data(shape = [None, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 3])

    self.network = conv_2d(self.network, 64, 3, activation='relu')
    self.network = conv_2d(self.network, 64, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 128, 3, activation='relu')
    self.network = conv_2d(self.network, 128, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = conv_2d(self.network, 256, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = conv_2d(self.network, 512, 3, activation='relu')
    self.network = max_pool_2d(self.network, 2, strides=2)

    self.network = fully_connected(self.network, 4096, activation='relu')
    self.network = dropout(self.network, 0.5)
    self.network = fully_connected(self.network, 4096, activation='relu')
    self.network = dropout(self.network, 0.5)
    self.network = fully_connected(self.network, 5, activation='softmax')

    self.network = regression(self.network, optimizer='adam',
                              loss='categorical_crossentropy',
                              learning_rate=0.0001)
    self.model = tflearn.DNN(
      self.network,
      tensorboard_dir=constants.DATA_PATH,
      checkpoint_path =constants.DATA_PATH + '/gun_checkpoint',
      max_checkpoints = 1,
      tensorboard_verbose = 2
    )
    self.load_model()
    print('-----------------------Model Loaded----------------------')

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('----------------Dataset found and loaded-----------------')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('--------------------Training network----------------------')
    print('Images validation:'+str(len(self.dataset._images_validation)))
    print('Labels Validation:'+str(len(self.dataset._labels_validation)))
    print ('Images training'+str(len (self.dataset._images)))
    print ('Labels training'+str(len (self.dataset._labels)))
    self.model.fit(
      self.dataset._images, self.dataset._labels,
      validation_set = (self.dataset._images_validation, self.dataset._labels_validation),
      n_epoch = 20,
      batch_size = 50,
      shuffle = True,
      show_metric = True,
      snapshot_step = 10,
      snapshot_epoch = True,
      run_id = 'gun_cnn'
    )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, constants.IMAGE_HEIGHT, constants.IMAGE_WIDTH, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(constants.DATA_PATH, constants.MODEL_NAME))
    print('---------------Model trained and saved at ' + constants.MODEL_NAME + '-------------------')

  def load_model(self):
    if isfile(join(constants.DATA_PATH, constants.MODEL_NAME)):
      self.model.load(join(constants.DATA_PATH, constants.MODEL_NAME))
      print('---------------------Model loaded from ' + constants.MODEL_NAME + '------------------------')

network=gun_cnn()
network.start_training ()
print ('---------------------Network Trained-----------------------------')
network.save_model ()
