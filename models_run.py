# -*- coding: utf-8 -*-
"""
# **Sound classification in the presence of missing samples**
"""
# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import random 
from numpy import savez_compressed, load
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
"""Libraries for the LeaveOneGroupOut CV"""
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import os
from helper_functions import * 

args = get_args()

if(args.pretrained):
  data_file = "dataset16kHz.npz"
  print("You have chosen the pretrained model.")
else:
  data_file = "dataset8kHz.npz"
  print("You have chosen the CNN model.")

base_path = "./UrbanSound8K/dati/"
labels_file = "all_labels.csv"

data, labels = get_data(base_path, data_file, labels_file)

results = run_model(data, labels, corrupt_train=args.corruptTrain, corrupt_test=args.corruptTest, corruption_length=args.corruptionSize, multiple=args.multiple, pretrained=args.pretrained) 
save_results(results)