from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions,VGG16
from keras.models import Model
from pickle import dump
import pickle
import pandas as pd
import cv2
import numpy as np


model = pickle.load(open('finalized_model_224.sav', 'rb'))
#print(pre)
cs=pd.read_csv('C:/Users/DELL/Desktop/r2ps1/sample.csv')
labels=list(cs['label'])
image_name=list(cs['image'])

data = []

for i in range(len(image_name)):
    path='C:/Users/DELL/Desktop/r2ps1/test/'+image_name[i]
    print(path)
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        data.append(image)
    except:

        continue
data = np.array(data)/255.0

yhat=model.predict(data)
print(yhat)

predIdxs = np.argmax(yhat, axis=1)
print(predIdxs)



cs['label']=predIdxs
cs.to_csv('pred.csv')
