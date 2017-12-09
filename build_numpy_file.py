import constants
from os.path import join
import pandas as pd
import numpy as np
from PIL import Image
import pylab as pl
import sys
import cv2
import h5py

def gun_vector(x):
    d = np.zeros(len(constants.GUN_TYPE))
    d[x] = 1.0
    return d

data = pd.read_csv(join(constants.DATA_PATH,constants.DATA_FILE_NAME))
labels_training = []
images_training = []
labels_validation = []
images_validation = []

index = 1
count=1
error_data=0
validation_data_count=0;
training_data_count=0
print('----------Starting processing data--------------')
for index, row in data.iterrows():
    print("In Iteration:"+str(index))
    gun = gun_vector(row[1])
    try:
        image = Image.open(row[0]).resize((224,224),Image.ANTIALIAS)
        if image is not None:
            if not count > int (constants.TRAINING_DATA_PERCENTAGE * constants.TOTAL_DATASET_COUNT):
                images_training.append(np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0], 3))
                labels_training.append(gun)
                training_data_count=training_data_count+1
            else:
                images_validation.append(np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0], 3))
                labels_validation.append(gun)
                validation_data_count=validation_data_count+1
            count=count+1
        else:
            data=data.drop(index)
            error_data=error_data+1
        index += 1
    except OSError as err:
        error_data = error_data + 1
        print("OS error: {0}".format(err))
    except:
        error_data = error_data + 1
        print("Unexpected error:", sys.exc_info()[0])
print('----------Data processed and segregated in validation and training sets--------------')
print("Total Validation Images Count: "+str(validation_data_count))
print("Total Training Images Count: "+str(training_data_count))
print("Total Error Data: "+str(error_data))
print("Size of Training images list: "+(str(sys.getsizeof(images_training))))
print("Size of Validation images list: "+(str(sys.getsizeof(images_training))))

with h5py.File(constants.TRAINING_IMAGES_FILENAME, 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=images_training)
print('----------Training images saved--------------')

with h5py.File(constants.TRAINING_LABELS_FILENAME, 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=labels_training)
print('----------Training labels saved--------------')

with h5py.File(constants.VALIDATION_IMAGES_FILENAME, 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=images_validation)
print('----------Validation Images saved--------------')

with h5py.File(constants.VALIDATION_LABELS_FILENAME, 'w') as hf:
    hf.create_dataset("name-of-dataset",  data=labels_validation)
print('----------Validation labels saved--------------')

print('Images validation:'+str(len(images_validation)))
print('Labels Validation:'+str(len(labels_validation)))
print ('Images training'+str(len (images_training)))
print ('Labels training'+str(len (labels_training)))
