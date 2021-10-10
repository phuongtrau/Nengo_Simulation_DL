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

# train_images, train_labels, test_images, test_labels = export_data()
exp_i_data = export_data_exp_i("experiment-i")
datasets = {"Base":exp_i_data}
subjects = ["S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S12","S13"]

acc_per_so = []
loss_per_so = []
ls_train_full = subjects.copy()
for sub in ls_train_full:
    subjects.remove(sub)
    #### Load data train ####
    train_data = Mat_Dataset(datasets,["Base"],subjects)
    #### Load data test ####
    test_data = Mat_Dataset(datasets,["Base"],[sub])
    
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

    # checkpoint_filepath = 'Nengo_weight'

    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='max',
    #     save_best_only=True)

    if do_training:
        with nengo_dl.Simulator(net_train, minibatch_size=64,seed=0) as sim:
            # run training
            sim.compile(
                optimizer=tf.optimizers.Adam(0.001),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.sparse_categorical_accuracy],
            )
            sim.fit(
                {converter.inputs[inp]: train_data.samples},
                {converter.outputs[dense]: train_data.labels},
                validation_data=(
                    {converter.inputs[inp]: test_data.samples},
                    {converter.outputs[dense]: test_data.labels},
                ),
                epochs=10,
                #callbacks=[model_checkpoint_callback],
            )
            # save the parameters to file
            path_params = "SNN_PARAMS/SNN_BED_LOSO_NON_{}".format(sub)
            sim.save_params(path_params)
        
    def run_network(
        activation,
        model,test_images,test_labels,
        params_file=path_params,
        n_steps=120,
        scale_firing_rates=5,
        synapse=None,
        n_test=400,
        ):
        #inp=tf.keras.layers.Input(shape=(28,28,1))
        # convert the keras model to a nengo network
        nengo_converter = nengo_dl.Converter(
            model=model,
            swap_activations={tf.nn.relu: activation},
            scale_firing_rates=scale_firing_rates,
            synapse=synapse,
        )

        # get input/output objects
        nengo_input = nengo_converter.inputs[model.inputs]
        nengo_output = nengo_converter.outputs[model.outputs]
        #with nengo_converter.net:
            #nengo_dl.configure_settings(simplifications=[])
        

        # add a probe to the first convolutional layer to record activity.
        # we'll only record from a subset of neurons, to save memory.
        sample_neurons = np.linspace(
            0,
            np.prod(conv0.shape[1:]),
            1000,
            endpoint=False,
            dtype=np.int32,
        )
        with nengo_converter.net:
            conv0_probe = nengo.Probe(nengo_converter.layers[conv0][sample_neurons])
            
    
        # repeat inputs for some number of timesteps
        tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

        # set some options to speed up simulation
        with nengo_converter.net:
            nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)
            nengo_dl.configure_settings(stateful=False)

        # build network, load in trained weights, run inference on test images
        with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=10, progress_bar=False
        ) as nengo_sim:
            params = list(nengo_sim.keras_model.weights)
            print(len(params))
            nengo_sim.load_params(params_file)
            data = nengo_sim.predict({nengo_input: tiled_test_images})

        # compute accuracy on test data, using output of network on
        # last timestep
        predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
        accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
        print("DO CHINH XAC:", accuracy)

    run_network(model=model,test_labels=test_data.labels,\
            test_images=test_data.samples,activation=nengo.RectifiedLinear())

    for s in [ 0.005, 0.01]:
        for scale in [1, 2, 5, 10, 20, 30, 40, 50, 100]:
            print(f"Synapse={s:.3f}", f"scale_firing_rates={scale:.3f}")
            run_network(
            activation=nengo.SpikingRectifiedLinear(),
            model=model,test_labels=test_labels,test_images=test_images,
            scale_firing_rates=scale,
            n_steps=120,
            synapse=s,
            )
            print('\n')
            print('THE ENERGY OF SNN WITH SCALE ', scale)
            energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 64, 32,1))*scale)
            energy.summary(
                columns=(
            "name",
            "synop_energy loihi",
            "neuron_energy loihi",
            "energy loihi",
            "energy cpu",
            "energy gpu",
            "synop_energy cpu",
            "synop_energy gpu",
            "neuron_energy cpu",
            "neuron_energy gpu"
                ),
                print_warnings=False,
            )
