import numpy as np

import cv2
from PIL import Image
import pylab as pl

image=Image.open('F:\\GUNS\Bullpup\AKMSU-original.jpg')
print(image.height)
print(image.width)
image=image.resize((224,224),Image.ANTIALIAS)

#image=image.crop((10,10,image.width*0.9,image.height*0.9));
print(image.height)
print(image.width)
a=np.asarray(image)
pl.imshow(a)
pl.show()

#print(image.size[1])
            #print(image.size[0])
            #a = np.asarray(image)
            #pl.imshow(a)
            #pl.show()
            #a = np.asarray(image)

