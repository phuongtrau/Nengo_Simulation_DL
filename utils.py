import numpy as np
from numpy import newaxis
import random
import os
#import PIL
#from PIL import ImageOps, Image
#import matplotlib.pyplot as plt
from scipy import ndimage
# from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow as tf

import nengo_dl
import cv2
#from torchvision.transforms import ToPILImage

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

position_i = ["justAPlaceholder", "symbol_1", "symbol_2",
              "symbol_3", "symbol_4", "symbol_5",
              "symbol_6", "symbol_7", "symbol_8",
              "symbol_9", "symbol_10", "symbol_11",
              "symbol_12", "symbol_13", "symbol_14",
              "symbol_15", "symbol_16", "symbol_17"]

def token_position(x):

    return int(x.split('_')[-1]) - 1

def export_data(data_dir_test='experiment-i_hoang/test',
                data_dir_train='experiment-i_hoang/train',
                preprocess=True):
    
    #data_dict = dict()
    
    data_train = None
    labels_train = None
    data_test = None
    labels_test = None

    for _, dirs_train, _ in os.walk(data_dir_train):
        for directory_train in dirs_train:
            # each directory is a subject
            subject_train = directory_train
            
            for _, _, files_train in os.walk(os.path.join(data_dir_train, directory_train)):
                for file_train in files_train:
                    file_path_train = os.path.join(data_dir_train, directory_train, file_train)
                    # print(file_path_train)
                    with open(file_path_train, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                            raw_data_train = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            # Change the range from [0-1000] to [0-255].
                            file_data_train = np.round(raw_data_train * 255 / 1000).astype(np.uint8)
                            file_data_train = file_data_train.reshape(1, 1, 2048)

                            # Turn the file index into position list,
                            # and turn position list into reduced indices.
                            file_label_train = token_position(position_i[int(file_train[:-4])])
                            file_label_train = np.array([file_label_train])
                            file_label_train = file_label_train.reshape(1,1,1)

                            if data_train is None:
                                data_train = file_data_train
                            else:
                                data_train = np.concatenate((data_train, file_data_train), axis=0)
                            if labels_train is None:
                                labels_train = file_label_train
                            else:
                                labels_train = np.concatenate((labels_train, file_label_train), axis=0)

    for _, dirs_test, _ in os.walk(data_dir_test):
        for directory_test in dirs_test:
            # each directory is a subject
            subject_test = directory_test
            for _, _, files_test in os.walk(os.path.join(data_dir_test, directory_test)):
                for file_test in files_test :
                    file_path_test = os.path.join(data_dir_test, directory_test, file_test)
                    with open(file_path_test, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                            raw_data_test = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            # Change the range from [0-1000] to [0-255].
                            file_data_test = np.round(raw_data_test * 255 / 1000).astype(np.uint8)
                            file_data_test = file_data_test.reshape(1, 1, 2048)

                            # Turn the file index into position list,
                            # and turn position list into reduced indices.
                            file_label_test = token_position(position_i[int(file_test[:-4])])
                            file_label_test = np.array([file_label_test])
                            file_label_test = file_label_test.reshape(1,1,1)

                            if data_test is None:
                                data_test = file_data_test
                            else:
                                data_test = np.concatenate((data_test, file_data_test), axis=0)
                            if labels_test is None:
                                labels_test = file_label_test
                            else:
                                labels_test = np.concatenate((labels_test, file_label_test), axis=0)

            #data_dict[subject] = (data, labels)

    return data_train, labels_train , data_test, labels_test

<<<<<<< HEAD
def run_network(
    activation,
    model,test_images,test_labels,
    params_file="SNN_BED_LOSO_NON_PRE_1",
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
        np.prod(model.layers[1].shape[1:]),
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
=======
>>>>>>> 1c6b7f211b8fbe697096a4755d0f347c21da7ba7
