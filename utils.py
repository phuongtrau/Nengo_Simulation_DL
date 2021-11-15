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

positions_i_short = ["justAPlaceholder", "supine", "right",
               "left", "right", "right",
               "left", "left", "supine",
               "supine", "supine", "supine",
               "supine", "right", "left",
               "supine", "supine", "supine"]

def token_position_short(x):
  return {
      'supine': 0,
      'left': 1,
      'right': 2,
  }[x]

def token_position(x):

    return int(x.split('_')[-1]) - 1

def token_position_new(x):
  return {
      "symbol_1":0, 
      "symbol_2":1,
      "symbol_3":2, 
      "symbol_4":3, 
      "symbol_5":4,
      "symbol_6":5, 
      "symbol_7":6, 
      "symbol_8":7,
      "symbol_9":8, 
      "symbol_10":9, 
      "symbol_11":10,
      "symbol_12":11, 
      "symbol_13":12, 
      "symbol_14":13,
      "symbol_15":14, 
      "symbol_16":15, 
      "symbol_17":16
  }[x]


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

def export_data_exp_i(path='experiment-i_hoang/train',preprocess=True):
    
    data_dict = dict()
    
    # data_train = None
    # labels_train = None
    # data_test = None
    # labels_test = None

    dataset = {}

    for _, dirs, _ in os.walk(path):
        for directory in dirs:
        # each directory is a subject
            subject = directory
            data = None
            labels = None
            max_val = []
            for _, _, files in os.walk(os.path.join(path, directory)):
                # print(files)
                for file in files:
                # print(file)
                    file_path = os.path.join(path, directory, file)
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                                        
                            raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            
                            if preprocess is True:
                                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                                
                                # Spatio-temporal median filter 3x3x3
                                raw_data = ndimage.median_filter(raw_data, 3)
                                past_image = ndimage.median_filter(past_image, 3)
                                future_image = ndimage.median_filter(future_image, 3)
                                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                                raw_data = np.median(raw_data, axis=0)
                                    
                                    # Change the range from [0-1000] to [0-255].
                            file_data = np.round(raw_data*255/1000).astype(np.uint8)            
                            
                            # file_data = cv2.equalizeHist(file_data)

                            file_data = file_data.reshape(1,1,2048)

                            

                                        # Turn the file index into position list,
                                        # and turn position list into reduced indices.
                            file_label= token_position_new(position_i[int(file[:-4])])
                                        
                            file_label = np.array([file_label])
                                        
                            file_label = file_label.reshape(1,1,1)

                            if data is None:
                                data = file_data
                            else:
                                data = np.concatenate((data, file_data), axis=0)
                            if labels is None:
                                labels = file_label
                            else:
                                labels = np.concatenate((labels, file_label), axis=0)
            dataset[subject] = (data, labels)

    return dataset

def export_data_exp_i_3_classes(path='experiment-i_hoang/train',preprocess=True):
    
    data_dict = dict()
    
    # data_train = None
    # labels_train = None
    # data_test = None
    # labels_test = None

    dataset = {}

    for _, dirs, _ in os.walk(path):
        for directory in dirs:
        # each directory is a subject
            subject = directory
            data = None
            labels = None
            max_val = []
            for _, _, files in os.walk(os.path.join(path, directory)):
                # print(files)
                for file in files:
                # print(file)
                    file_path = os.path.join(path, directory, file)
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                                        
                            raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            
                            if preprocess is True:
                                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                                
                                # Spatio-temporal median filter 3x3x3
                                raw_data = ndimage.median_filter(raw_data, 3)
                                past_image = ndimage.median_filter(past_image, 3)
                                future_image = ndimage.median_filter(future_image, 3)
                                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                                raw_data = np.median(raw_data, axis=0)
                                    
                                    # Change the range from [0-1000] to [0-255].
                            file_data = np.round(raw_data*255/1000).astype(np.uint8)            
                                        
                            file_data = file_data.reshape(1,1, 2048)
                                        # Turn the file index into position list,
                                        # and turn position list into reduced indices.
                            file_label= token_position_short(positions_i_short[int(file[:-4])])
                                        
                            file_label = np.array([file_label])
                                        
                            file_label = file_label.reshape(1,1,1)

                            if data is None:
                                data = file_data
                            else:
                                data = np.concatenate((data, file_data), axis=0)
                            if labels is None:
                                labels = file_label
                            else:
                                labels = np.concatenate((labels, file_label), axis=0)
            dataset[subject] = (data, labels)

    return dataset

def export_data_exp_i_cnn(path='experiment-i',preprocess=True):
    
    data_dict = dict()
    
    # data_train = None
    # labels_train = None
    # data_test = None
    # labels_test = None

    dataset = {}

    for _, dirs, _ in os.walk(path):
        for directory in dirs:
        # each directory is a subject
            subject = directory
            data = None
            labels = None
            max_val = []
            for _, _, files in os.walk(os.path.join(path, directory)):
                # print(files)
                for file in files:
                # print(file)
                    file_path = os.path.join(path, directory, file)
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                                        
                            raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            
                            if preprocess is True:
                                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                                
                                # Spatio-temporal median filter 3x3x3
                                raw_data = ndimage.median_filter(raw_data, 3)
                                past_image = ndimage.median_filter(past_image, 3)
                                future_image = ndimage.median_filter(future_image, 3)
                                raw_data = np.concatenate((raw_data[np.newaxis, :, :], past_image[np.newaxis, :, :], future_image[np.newaxis, :, :]), axis=0)
                                raw_data = np.median(raw_data, axis=0)
                                    
                                    # Change the range from [0-1000] to [0-255].
                            file_data = np.round(raw_data*255/1000).astype(np.uint8)            
                                        
                            file_data = file_data.reshape((1,64,32))
                                        # Turn the file index into position list,
                                        # and turn position list into reduced indices.
                            file_label= token_position_new(position_i[int(file[:-4])])
                                        
                            file_label = np.array([file_label])
                                        
                            # file_label = file_label.reshape(1,1)

                            if data is None:
                                data = file_data
                            else:
                                data = np.concatenate((data, file_data), axis=0)
                            if labels is None:
                                labels = file_label
                            else:
                                labels = np.concatenate((labels, file_label), axis=0)
            dataset[subject] = (data, labels)

    return dataset

import cv2 

class Mat_Dataset():
  def __init__(self,datasets, mats, Subject_IDs):

    self.samples = []
    self.labels = []

    for mat in mats:
      data = datasets[mat]
      self.samples.append(np.concatenate([data.get(key)[0] for key in Subject_IDs],axis=0))
      self.labels.append(np.concatenate([data.get(key)[1] for key in Subject_IDs],axis=0))

    self.samples = np.vstack(self.samples)
    self.labels = np.hstack(self.labels)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, idx):
    return self.samples[idx], self.labels[idx]

class Mat_Dataset_CNN():
  def __init__(self,datasets, mats, Subject_IDs):

    self.samples = []
    self.labels = []

    for mat in mats:
      data = datasets[mat]
      self.samples.append(np.vstack([data.get(key)[0] for key in Subject_IDs]))
      self.labels.append(np.hstack([data.get(key)[1] for key in Subject_IDs]))

    self.samples = np.vstack(self.samples)
    self.labels = np.hstack(self.labels)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, idx):
    return self.samples[idx], self.labels[idx]
