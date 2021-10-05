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
# sub_test = ['S8']
# ls_sub = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13']
# ls_sub.remove(sub_test)

# source = 'experiment-i_hoang'

train_images, train_labels, test_images, test_labels = export_data()



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

converter = nengo_dl.Converter(model)
net_train = converter.net

with net_train:
        # Disable the NengoDL graph optimizer
    nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)

do_training = True

checkpoint_filepath = 'Nengo_weight'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

if do_training:
    with nengo_dl.Simulator(net_train, minibatch_size=64,seed=0) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[inp]: train_images},
            {converter.outputs[dense]: train_labels},
            validation_data=(
                {converter.inputs[inp]: test_images},
                {converter.outputs[dense]: test_labels},
            ),
            epochs=10,
            #callbacks=[model_checkpoint_callback],
        )

        # save the parameters to file
        sim.save_params("SNN_PARAMS/SNN_BED_LOSO_NON_PRE_1")

run_network(model=model,test_labels=test_labels,test_images=test_images,activation=nengo.RectifiedLinear())

for s in [ 0.005, 0.01]:
    for scale in [1, 2, 5, 10, 20, 30, 40, 50, 100]:
        print(f"Synapse={s:.3f}", f"scale_firing_rates={scale:.3f}")
        run_network(
        activation=nengo.SpikingRectifiedLinear(),
        scale_firing_rates=scale,
        n_steps=120,
        synapse=s,
        )
    plt.show()