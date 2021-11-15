import numpy as np
from numpy import newaxis
import random
import os
#import PIL
#from PIL import ImageOps, Image
#import matplotlib.pyplot as plt
from scipy import ndimage
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow as tf

import nengo_dl
import cv2
from utils import *
import keras_spiking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# train_images, train_labels, test_images, test_labels = export_data()
exp_i_data = export_data_exp_i_cnn("experiment-i",preprocess=False)
datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]

acc_per_so = []
# loss_per_so = []
ls_train_full = subjects.copy()
for sub in ls_train_full:
    subjects.remove(sub)
    #### Load data train ####
    train_data = Mat_Dataset_CNN(datasets,["Base"],subjects)
    #### Load data test ####
    test_data = Mat_Dataset_CNN(datasets,["Base"],[sub])

    # y_train = to_categorical(train_data.labels, 17)
    # y_test = to_categorical(test_data.labels, 17)

    # print("train_images_shape",train_data.samples.shape)
    # print("test_images_shape",test_data.samples.shape)
    # print("train_labels_shape",y_train.shape)
    # print("test_labels_shape",y_test.shape)

    inp=tf.keras.layers.Input(shape=(64,32,1))

    conv0 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same", activation=tf.nn.relu)(inp)
    pool0 = tf.keras.layers.AveragePooling2D(pool_size=2,strides=2)(conv0)

    conv1 = tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding="same", activation=tf.nn.relu)(pool0)
    pool1 = tf.keras.layers.AveragePooling2D(pool_size=2,strides=2)(conv1)   

    conv2 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding="same", activation=tf.nn.relu)(pool1)
    conv3 = tf.keras.layers.Conv2D(filters=128,kernel_size=1,padding="same", activation=tf.nn.relu)(conv2)
    conv4 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding="same", activation=tf.nn.relu)(conv3)   
    pool2 = tf.keras.layers.AveragePooling2D(pool_size=2,strides=2)(conv4)

    conv5 = tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding="same", activation=tf.nn.relu)(pool2)   
    global_average_2D = tf.keras.layers.GlobalAvgPool2D()(conv5)
    flatten = tf.keras.layers.Flatten()(global_average_2D)
    dense = tf.keras.layers.Dense(units=17)(flatten)
    model = tf.keras.Model(inputs = inp, outputs = dense)

    model.compile(optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],)

    print(f'Training for subject {sub} ...')

    model.fit(train_data.samples, train_data.labels,
            batch_size=64,
            epochs=10,
            verbose=1,)
    model.summary() 

#     model.save("CNN_params_temp/{}-params".format(sub))
#     score = model.evaluate(test_data.samples, test_data.labels, verbose=0)
  
#     acc_per_so.append(score[1] * 100)
#     print("ACC: ",score[1] * 100,"%")
#     # loss_per_so.append(score[0])

#     subjects = ls_train_full.copy()

# print('------------------------------------------------------------------------')
# print('Score per subject out')
# for i in range(0, len(acc_per_so)):
#   print('------------------------------------------------------------------------')
#   print(f'> Subject {i+1} - Accuracy: {acc_per_so[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all subject out:')
# print(f'> Accuracy: {np.mean(acc_per_so)} (+- {np.std(acc_per_so)})')
# print('------------------------------------------------------------------------')