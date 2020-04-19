from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import pandas as pd
import glob
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')



dir='C:/Users/DELL/Desktop/r2ps1/train/'

data = []
cnt=0
for i in range(2):
    path=dir+str(i)+'/*'
    for path in glob.glob(path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        data.append(image)


import numpy as  np
x = np.array(data)
print(x.shape)
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='2', save_prefix='img', save_format='jpeg'):
    i += 1
    if i > 2000:
        break  # otherwise the generator would loop indefinitely