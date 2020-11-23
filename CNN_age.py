import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import tqdm
import tensorflow as tf
import gc

from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback
#wandb.init(project="age-classifier")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

filepath="model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy',save_weights_only=False, verbose=1, save_best_only=True, mode='max')

img_size = 100

def return_folder_info(textfile):
    global none_count
    # one big folder list
    folder = []
    # start processing metadata txt
    with open(textfile) as text:
        lines = text.readlines()
        for line in lines[1:]:
            line = line.strip().split("\t")
            # image path
            img_path = line[0]+"/"+prefix+line[2]+"."+line[1]
            if line[3] == "None":
                none_count += 1
                continue
            else:
                # We store the metadata info
                folder.append([file_path+img_path]+line[3:5])
                if folder[-1][1] in classes_to_fix:
                    folder[-1][1] = classes_to_fix[folder[-1][1]]
    return folder

# Methods for processing img arrays and one-hot generation
def imread(path, width, height):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def build_one_hot(age):
    label = np.zeros(len(classes), dtype=int)
    label[classes.index(age)] = 1
    return label

file_path = "./aligned/"
prefix = "landmark_aligned_face."
metadata = [file_path+'fold_0_data.txt',file_path+'fold_1_data.txt',file_path+'fold_2_data.txt',file_path+'fold_3_data.txt',file_path+'fold_4_data.txt']
classes = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(25, 32)", "(38, 43)", "(48, 53)", "(60, 100)"]

classes_to_fix = {'35': classes[5], '3': classes[0], '55': classes[7], '58': classes[7], 
'22': classes[3], '13': classes[2], '45': classes[5], '36': classes[5], 
'23': classes[4], '57': classes[7], '56': classes[6], '2': classes[0], 
'29': classes[4], '34': classes[4], '42': classes[5], '46': classes[6], 
'32': classes[4], '(38, 48)': classes[5], '(38, 42)': classes[5], '(8, 23)': classes[2],
 '(27, 32)': classes[4]}

none_count = 0

all_folders = []
for textfile in metadata:
    folder = return_folder_info(textfile)
    all_folders.append(folder)

all_data = []
all_labels = []
print("Start reading images data...")
for folder in all_folders:
    data = []
    labels = []
    for i in tqdm.tqdm(range(len(folder))):    #using tqdm to monitor progress
        img = imread(folder[i][0], img_size,img_size)
        one_hot = build_one_hot(folder[i][1])
        data.append(img)
        labels.append(one_hot)
    all_data.append(data)
    all_labels.append(labels)
    print("One folder done...")
print("All done!")

print("Appending Datas")
# Merge data and labels first
merged_data = np.concatenate((all_data[0],all_data[1],all_data[2],all_data[3],all_data[4]))
merged_labels = np.concatenate((all_labels[0],all_labels[1],all_labels[2],all_labels[3],all_labels[4]))

del prefix,file_path,metadata,classes,classes_to_fix,none_count,all_folders,folder,all_data,all_labels,data,labels,img,one_hot
gc.collect

val = []
for i in merged_labels:
    val.append(np.argmax(i))

value=[]
for i in val:
    if i < 3:
        value.append(0)
    elif i == 3:
        value.append(1)        
    elif i >= 4 and i < 7:
        value.append(2)        
    else:
        value.append(3)

print("==================Reading Augmentated Images=================")
path_2 = "./adolescent_aug/"
read_2 = os.listdir(path_2)
path_4 = "./elderly_aug/"
read_4 = os.listdir(path_4)

img_2 = []
img_4 = []
for img in tqdm.tqdm(read_2):
    img_2.append(imread(path_2+img, img_size, img_size))
    
for img in tqdm.tqdm(read_4):
    img_4.append(imread(path_4+img, img_size, img_size))

img_label2 = np.ones(len(img_2))*1
img_label4 = np.ones(len(img_4))*3

img_data = img_2 + img_4
val = np.concatenate((value, img_label2, img_label4))

img_data = np.array(img_data)
merged_data = np.concatenate((merged_data,img_data))

merged_labels = tf.keras.utils.to_categorical(val, 4)
merged_data = merged_data / 255

del val, img_data, value, img_label2, img_label4, read_4, path_4, read_2, path_2, img_2, img_4
gc.collect()

sweep_config = {
    'name': "Sweep_Tuning",
    'method': 'random',
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
#         'channel_1': {
#             'values': [16,32,64]
#         },
#         'channel_2': {
#             'values': [64,128,256]
#         },
#         'channel_3': {
#             'values': [256,512]
#         },
#         'dense_1': {
#             'values': [256,512]
#         },
#         'dense_2': {
#             'values': [32,64,128,256]
#         },
        'dropout': {
            'values': [0.3,0.4,0.5]
        },
        'kernel_size': {
            'values': [2,3,4,5]
        },
        'learning_rate': {
            'values': [1e-4,3e-4,3e-5,1e-5]
        },
        'weight_decay': {
            'values': [1e-12,1e-9,1e-6]
        },
        'test_split':{
            'values': [0.2,0.25,0.3]
        },
        'random_state':{
            'values': [11,20,88]
        },
        'opt':{
            'values': ["Adam", "RMSprop"]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="age-classifier")

def train():
    config_defaults = {
        'epochs': 20,
        'batch_size': 64,
        'weight_decay': 1e-3,
        'channel_1': 64,
        'channel_2': 128,
        'channel_3': 512,
        'dense_1': 256,    #256
        'dense_2': 128,    #128
        'dense_3': 4,
        'learning_rate': 3e-4,
        'kernel_size': 5,
        'dropout': 0.8,
        'test_split': 0.3,
        'random_state': 88,
        'opt': "Adam"
    }
    wandb.init(config=config_defaults) #, project="age-classifier")
    config = wandb.config

    print("==============Splitting Data===========")
    X_train, X_test, Y_train, Y_test = train_test_split(merged_data, merged_labels, test_size=config.test_split, random_state=config.random_state)

    model = Sequential()
    model.add(Conv2D(config.channel_1, kernel_size=config.kernel_size, padding='same', activation='relu', input_shape=(img_size, img_size, 3), kernel_regularizer=l2(config.weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(config.channel_2, kernel_size=config.kernel_size, padding='same', activation='relu', kernel_regularizer=l2(config.weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(config.channel_3, kernel_size=config.kernel_size, padding='same', activation='relu', kernel_regularizer=l2(config.weight_decay)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(config.dense_1, kernel_regularizer=l2(config.weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(config.dropout))

    model.add(Dense(config.dense_2, kernel_regularizer=l2(config.weight_decay)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(config.dropout))

    model.add(Dense(config.dense_3, activation="softmax", kernel_regularizer=l2(config.weight_decay)))
    
    if config.opt == "Adam":
        optimizer = Adam(lr=config.learning_rate)
    elif config.opt == "RMSprop":
        optimizer = RMSprop(lr=config.learning_rate)

    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=config.epochs, batch_size=config.batch_size, verbose=1, callbacks=[checkpoint,WandbCallback()])
    #return model

model = train()
#wandb.agent(sweep_id, train)


#==============IMAGE AUGMENTATION=============#
#train_datagen2 =  ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# train_datagen4 = train_datagen2

# train_datagen2.fit(val_data2)
# train_datagen4.fit(val_data4)

# train_generator2 = train_datagen2.flow(
#     val_data2, val2,
#     batch_size=4,
#     save_to_dir="adolescent_aug",
#     save_prefix="adolescent",
# )
# train_generator4 = train_datagen4.flow(
#     val_data4, val4,
#     batch_size=8,
#     save_to_dir="elderly_aug",
#     save_prefix="elderly",
# )
#==========END IMAGE AUGMENTATION=============#

