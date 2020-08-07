#%%
import cv2
import numpy as np
import tensorflow as tf  
from tensorflow.keras.models import load_model

model= load_model("Mask-Detection-CNN-")

#for testing
def valImg(filepath):
    IMG_SIZE= 80
    img_array= cv2.imread(filepath)
    new_array= cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)


prediction= model.predict([valImg('Dataset/Validation/1374.png')])
print(prediction)