from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import os
import pandas as pd

datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


cs=pd.read_csv('C:/Users/DELL/Desktop/r2ps1/train_label.csv')

print(cs.head())
labels=list(cs['label'])
image_name=list(cs['image'])
print(len(image_name))




data = []

for i in range(500):
    path='C:/Users/DELL/Desktop/r2ps2/train/2/'+str(i)+'.jpg'
    print(path)
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        data.append(image)
    except:

        continue
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