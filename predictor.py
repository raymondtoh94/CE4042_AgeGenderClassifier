#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[2]:


import logging
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import *
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
import os
tf.get_logger().setLevel(logging.ERROR)


# In[57]:


def load_img(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255
    img_array = img_array.reshape(-1,100, 100, 3)

    plt.imshow(img)
    plt.show()
    return img_array#img_gender, img_smile


# In[58]:


def merge(image, gender_model, smile_model,age_model):
  # input_shape = (100,100,3)
    gender_pred = gender_model.predict(image)
    smile_pred = smile_model.predict(image)
    age_pred = age_model.predict(image)

    if gender_pred < 0.5:
        print("Female", gender_pred)
    else:
        print("Male", gender_pred)

    if smile_pred < 0.5:
        print("Not Smiling",smile_pred)
    else:
        print("Smiling",smile_pred)


    print(f"Predicted array - {age_pred}")
    age_pred = np.argmax(age_pred)
    print(f"Predicted Age Group - {label[str(age_pred)]}")


# In[109]:


gender_model = load_model('model_93_percent_gender_size100.h5')
smiling_model = load_model('model_87_percent_smiling.h5')
age_model = load_model('model.h5')
img_path = 'landmark_aligned_face.2282.11598278104_e2ab7edede_o.jpg'

label = {}
label['0'] = "Children"
label['1'] = "Adolescent"
label['2'] = "Adult"
label['3'] = "Elderly"


# In[110]:


img_arr = load_img(img_path)
merge(img_arr, gender_model, smiling_model, age_model)


# In[ ]:





# In[ ]:




