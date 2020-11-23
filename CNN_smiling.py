#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[36]:


get_ipython().system("tar xvzf '/content/drive/MyDrive/UTKFace.tar.gz'")


# In[ ]:


get_ipython().system('pip install --upgrade wandb')
import wandb
from wandb.keras import WandbCallback
wandb.init(project="Make me Happy or Sad?")

# wandb login 205605454fb591482f8baaea600c6536ae5de8a6


# In[ ]:


get_ipython().system('pip install -U keras-tuner')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd
import cv2
import matplotlib.pyplot as plt


import os
from sklearn.metrics import f1_score
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D, Reshape, Activation
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
# from kerastuner.tuners import RandomSearch
# from kerastuner.engine.hyperparameters import HyperParameters
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
import time
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K


# In[ ]:


IMG_WIDTH_HEIGHT = 200
TRAINING_SAMPLES =5000
VALIDATION_SAMPLES = 1500
TEST_SAMPLES = 1500
BATCH_SIZE = 16
NUM_EPOCHS = 20
classifier = 'Smiling'


# In[ ]:


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

df = pd.read_csv("/content/drive/MyDrive/list_attr_celeba.csv")
df = reduce_mem_usage(df)

df.reset_index(inplace = True)
df.set_index('image_id', inplace=True)
df.drop(columns=['index'], inplace=True)
df.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
df.shape

filename = df.index[567]
image_folder = "/content/img_align_celeba/"
imagepath = image_folder + filename
img = cv2.imread(imagepath)

import gc
del img
gc.collect()

tvtsplit = pd.read_csv("/content/drive/MyDrive/list_eval_partition.csv")
tvtsplit = reduce_mem_usage(tvtsplit)

tvtsplit['partition'].value_counts().sort_index()

tvtsplit.reset_index(inplace=True)
tvtsplit.set_index("image_id", inplace=True)
tvtsplit.drop(columns=['index'], inplace=True)

tvtmerge = pd.merge(tvtsplit, df[classifier], on="image_id", how="inner")
tvtmerge = reduce_mem_usage(tvtmerge)

del tvtsplit
gc.collect()


def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = cv2.resize(x, (IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT))
    #x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test
    
    '''
    
    df_ = tvtmerge[(tvtmerge['partition'] == partition) 
                           & (tvtmerge[attr] == 0)].sample(int(num_samples/2))
    
    #df_ = df_.append(tvtmerge[(tvtmerge['partition'] == partition) & (tvtmerge[attr] == 1)].sample(int(num_samples/2)))
    df_ = pd.concat([df_,
                      tvtmerge[(tvtmerge['partition'] == partition) 
                                  & (tvtmerge[attr] == 1)].sample(int(num_samples/2))])
    df_ = reduce_mem_usage(df_)
    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(image_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, 3)
        y_ = np.array(df_[attr])
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(image_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT)).astype(np.float32) / 255.0
            #im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])
    del df_
    gc.collect()
    return x_, y_




# In[ ]:


# Train data
x_train, y_train = generate_df(0, classifier, TRAINING_SAMPLES)

# Train - Data Preparation - Data Augmentation with generators
train_datagen =  ImageDataGenerator(
  #preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
x_train, y_train,
batch_size=BATCH_SIZE
)

del x_train, y_train
gc.collect()

x_valid, y_valid = generate_df(1, classifier, VALIDATION_SAMPLES)


# # Model

# In[ ]:


# def build_model(hp):  # random search passes this hyperparameter() object 
#     model = Sequential()
    
#     model.add(Conv2D(hp.Int('input_units',
#                                 min_value=32,
#                                 max_value=256,
#                                 step=32), (3, 3), input_shape=(IMG_WIDTH_HEIGHT,IMG_WIDTH_HEIGHT,3)))
    
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(4, 4)))
#     model.add(BatchNormalization())
#     for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
#         model.add(Conv2D(hp.Int(f'conv_{i}_units',
#                                 min_value=32,
#                                 max_value=256,
#                                 step=32), (2, 2)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#     model.add(Flatten()) 
#     model.add(Dense(512))
#     model.add(Activation("relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(512))
#     model.add(Activation("relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(1))
#     model.add(Activation("sigmoid"))
#     model.compile(optimizer="adam",
#                   loss="binary_crossentropy",
#                   metrics=["accuracy"])
    
#     return model

def build_model(IMG_WIDTH_HEIGHT, classes):
  model = Sequential()
  inputShape = (IMG_WIDTH_HEIGHT, IMG_WIDTH_HEIGHT, 3)
  chanDim = -1

  #Input Layer
  model.add(Conv2D(192, (3,3), padding="same", input_shape=inputShape))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(4,4)))

  #Conv layer
  model.add(Conv2D(192, (3,3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))

  #Conv Layer
  model.add(Conv2D(160, (2,2), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))

  #Conv Layer  Extra one (gave 92%)
  model.add(Conv2D(128, (2,2), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))

  #Conv Layer  Extra one (Use this!)
  model.add(Conv2D(96, (2,2), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))

  #Conv Layer  Extra one (Use this!)
  model.add(Conv2D(64, (2,2), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))

  #Conv Layer
  model.add(Conv2D(32, (2,2), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  #Conv Layer
  model.add(Conv2D(16, (2,2), padding="same")) # default is 32
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Flatten()) 
  model.add(Dense(512))
  model.add(Activation("relu"))
  #model.add(Dropout(0.5))
  model.add(Dense(512))
  model.add(Activation("relu"))
  #model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation("sigmoid"))

  model.compile(loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])
  model.summary()
  return model


# In[ ]:


# LOG_DIR = f"{int(time.time())}"
# tensorboard = TensorBoard(log_dir=LOG_DIR)

filepath="model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',save_weights_only=False, verbose=1, save_best_only=True, mode='max')


# tuner = RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=5,  # how many variations on model?
#     executions_per_trial=2,  # how many trials per variation? (same model could perform differently)
#     directory=LOG_DIR)

# tuner.search_space_summary()

# tuner.search(train_generator,
#              epochs=NUM_EPOCHS,
#              verbose=2,
#              batch_size=BATCH_SIZE,
#              callbacks=[tensorboard,checkpoint],
#              validation_data=(x_valid, y_valid))

# tuner.results_summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model_ = build_model(IMG_WIDTH_HEIGHT, 2)


r = model_.fit(
  train_generator,
  validation_data=(x_valid, y_valid),
  epochs=NUM_EPOCHS,
  verbose =1,
  steps_per_epoch=(TRAINING_SAMPLES//BATCH_SIZE),
  # callbacks = [checkpoint,WandbCallback()]
  callbacks = [checkpoint]
)



# In[ ]:


# tuner.get_best_hyperparameters()[0].values


# In[ ]:


from keras.models import load_model

best_model = load_model('./model.h5')

x_test, y_test = generate_df(2, classifier, TEST_SAMPLES)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test.shape


### START CODE HERE ### (1 line)
preds = best_model.evaluate(x_test,y_test)
### END CODE HERE ###
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

Y_pred = best_model.predict(x_test)

predict_labels = Y_pred
predict_labels[predict_labels<0.5] = 0
predict_labels[predict_labels>=0.5] = 1

from sklearn.metrics import f1_score
print('f1_score:', f1_score(y_test, predict_labels))


# In[ ]:


# Plot loss function value through epochs
plt.figure(figsize=(18, 6))
plt.plot(r.history['loss'], label = 'train')
plt.plot(r.history['val_loss'], label = 'valid')
plt.legend()
plt.title('Loss Function')
# plt.savefig('InceptionV3_accuracy.png')
plt.show()

# Plot accuracy through epochs
plt.figure(figsize=(18, 6))
plt.plot(r.history['accuracy'], label = 'train')
plt.plot(r.history['val_accuracy'], label = 'valid')
plt.legend()
plt.title('Accuracy')
# plt.savefig('InceptionV3_accuracy.png')
plt.show()


# In[ ]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def load_img(img_path):
  img = image.load_img(img_path, target_size=(200, 200))
  img_array = image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_batch /= 255

  plt.imshow(img)
  plt.show()

  return img_batch


# In[ ]:


img_path = '/content/photo_2020-11-20_18-57-38.jpg'
img = load_img(img_path)
model = load_model('/content/model.h5')

pred = model.predict(img)

if pred < 0.5:
  print("Not Smiling")
else:
  print("Smiling")


# In[ ]:




