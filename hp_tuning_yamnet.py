# -*- coding: utf-8 -*-
"""
# Hyperparameters tuning - Bayesian optimization
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import random 
import os
import json 
from numpy import savez_compressed, load
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.optimizer import forest_minimize
from helper_functions import * 

# setting a random seed
SEED=42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC']='1'

data_file = "dataset16kHz.npz"
base_path = "./UrbanSound8K/dati/"
labels_file = "all_labels.csv"

data, labels = get_data(base_path, data_file, labels_file)
data = data.reshape(-1,64000,1)
yamnet = hub.load("https://tfhub.dev/google/yamnet/1") 

groups = np.empty(shape=data.shape[0], dtype=int)
groups.shape

groups[:873]=1
groups[873:1761]=2
groups[1761:2686]=3
groups[2686:3676]=4
groups[3676:4612]=5
groups[4612:5435]=6
groups[5435:6273]=7
groups[6273:7079]=8
groups[7079:7895]=9
groups[7895:]=10

def corrupt_data(fold, width, multiple_fragments=False):
  if multiple_fragments:
    corrupt_data_multiple_fragments(fold,width)
  else:
    for sample in fold: 
      rand1 = random.randint(0, int(len(sample)/1.3)) #24000 
      sample[rand1:rand1+width] = 0 
      zeros = np.zeros(shape=sample[rand1:].shape)
      if (sample[rand1:]==zeros).all() == True:
          rand1 = random.randint(0, int(len(sample)/6)) # 5000 
          sample[rand1:rand1+width] = 0
  return fold
  
def corrupt_data_multiple_fragments(fold,width):
  for sample in fold:
    # corruption divided into 4 fragments
    rand1 = random.randint(0, int(len(sample)/12))
    rand2 = random.randint(int(len(sample)/6.5), int(len(sample)/4.5))
    rand3 = random.randint(int(len(sample)/3.5), int(len(sample)/2.5))
    rand4 = random.randint(int(len(sample)/2), int(len(sample)/1.1))
    rands = [rand1, rand2, rand3, rand4]
    for rand in rands:    
        sample[rand:rand+int(width/4)] = 0 
  return fold

# YAMNet: range (-1,1); 16 kHz sr; Mono-channel
def get_embeddings_yamnet(training_set, test_set, yamnet): 
  print("Extracting embeddings..")
  scaler = MinMaxScaler(feature_range=(-1, 1))
  X_scaled = scaler.fit_transform(training_set.reshape(-1, 64000))
  embeddings = []
  for e in X_scaled:
    embeddings.append(yamnet(e)[1])
  embeddings = np.array(embeddings)
  embeddings = embeddings.reshape(-1, 8*1024)

  X_ts_scaled = scaler.transform(test_set.reshape(-1, 64000))
  embeddings_ts = []
  for e in X_ts_scaled:
    embeddings_ts.append(yamnet(e)[1])
  embeddings_ts = np.array(embeddings_ts)
  embeddings_ts = embeddings_ts.reshape(-1, 8*1024)
  gc.collect() 
  return embeddings, embeddings_ts

def get_classifier(ndn1, ndn2, ndn3, dropout, l2_regu):
    classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8192), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(2**ndn1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_regu)), 
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(2**ndn2, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_regu)), 
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(2**ndn3, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2_regu)), 
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(10, activation="softmax") 
    ], name='classifier')

    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=2, 
                                            restore_best_weights=True)
    return classifier

# define the valid search-dimensions for each parameter
dim_ndn1 = Integer(low=6, high=8, name="ndn1")
dim_ndn2 = Integer(low=5, high=6, name="ndn2")
dim_ndn3 = Integer(low=4, high=5, name="ndn3")
dim_dropout = Real(low=0.3, high=0.7, prior="log-uniform", name="dropout")
dim_l2_regu = Real(low=1e-3, high=1e-1, prior="log-uniform", name="l2_regu")

dimensions = [dim_ndn1, dim_ndn2, dim_ndn3, dim_dropout, dim_l2_regu]

default_parameters = [8, 6, 4, 0.3863790562689953, 0.001294791175975606]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=20,
    verbose=0,
    mode='auto',
    baseline=0.5,
    restore_best_weights=False
)

best_accuracy = 0
best_params = {}

# function to save the best parameters 
def save_params(params, accuracy):
  with open ("./best_params_yamnet.txt", 'w') as f:
    f.write("Accuracy: "+ str(accuracy))
    f.write(str(params))

@use_named_args(dimensions=dimensions)
def fitness(ndn1, ndn2, ndn3, dropout, l2_regu):
    """
    Hyper-parameters:
    l2_regu:           L2-Regularization term.
    dropout:           Dropout rate.
    ndn1:              Number of nodes in the first dense layer.
    ndn2:              Number of nodes in the second dense layer.
    ndn3:              Number of nodes in the third dense layer.
    """
    # Print the hyper-parameters
    print('ndn1:', 2**ndn1)
    print('ndn2:', 2**ndn2)
    print('ndn3:', 2**ndn3)
    print('l2:', l2_regu)
    print('droput:', dropout)
    print()

    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    import gc # garbage collector 
    logo = LeaveOneGroupOut()
    i=1
    scores = {} #'1': [1.35, 0.61]
    histories = {} 
    predicted_labels = {}
    '''
    corruption_length=500
    mode = "C_ts"
    model_name = "cnn"
    multiple=True
    '''
    cm_old = np.zeros((10,10))
    cm = np.zeros((10,10))
    for tr_idx, ts_idx in logo.split(data, labels, groups):
      model = get_classifier(ndn1=ndn1, ndn2=ndn2, ndn3=ndn3, dropout=dropout, l2_regu=l2_regu)
      print(" (TEST) FOLD: ", i)
      X_tr = data[tr_idx, :, :]
      y_tr = labels[tr_idx]
      X_ts = data[ts_idx, :, :]
      y_ts = labels[ts_idx]
      #X_ts_corrupted = corrupt_data(X_ts, corruption_length, multiple_fragments=multiple)
      X_tr, X_ts = get_embeddings_yamnet(X_tr, X_ts, yamnet)
      X_tr, X_vl, y_tr, y_vl = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, shuffle=True)
      history = model.fit(X_tr, y_tr, validation_data=(X_vl, y_vl), batch_size=128, epochs=30, callbacks=[early_stopping])
      if(early_stopping.stopped_epoch):
        print("Early stopping at epoch {} in fold {}. Time to stop!".format(early_stopping.stopped_epoch, i))
        accuracy = history.history['val_accuracy'][-1]
        return -accuracy
      # [0] = loss, [1] = accuracy (evaluate)
      score = model.evaluate(X_ts, y_ts)[1]
      predictions = model.predict(X_ts) #(873,10)
      predicted_labels_fold = predictions.argmax(axis=-1)
      predicted_labels[i] = predicted_labels_fold
      cm_old = confusion_matrix(y_ts, np.argmax(predictions,axis=1), normalize='true')
      cm+=cm_old
      scores[i]=score
      histories[i]=history
      i+=1
      del X_tr, X_ts, y_tr, y_ts, model
      gc.collect()

    # Get the classification accuracy on the test set

    accuracy = sum(scores.values())/len(scores)
    global best_accuracy
    global best_params
    if(accuracy > best_accuracy):
      best_accuracy = accuracy
      best_params = {'ndn1': ndn1, 'ndn2': ndn2, 'ndn3': ndn3, 'l2_regu': l2_regu, 'dropout': dropout}
    # Print the classification accuracy
    print()
    print("Mean accuracy CV: {0:.2%}".format(accuracy))
    print()

    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    save_params(best_params, best_accuracy)
    return -accuracy

# Sequential optimisation using decision trees
res=forest_minimize(fitness, dimensions, base_estimator="RF", n_calls=100, x0=default_parameters, initial_point_generator="lhs", kappa=4)