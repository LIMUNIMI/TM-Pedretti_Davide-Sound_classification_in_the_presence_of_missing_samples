# Importing libraries
import numpy as np
from numpy import load
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_hub as hub
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential 
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import random 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import os 
import gc
import argparse

# function to get arguments from the command line
def get_args():
  parser = argparse.ArgumentParser()
  # optional arguments
  parser.add_argument("--pretrained", help="True if you want to use YAMNet, False otherwise", action="store_true")
  parser.add_argument("--corruptionSize", help="Set the corruption size", type=int, default=500)
  parser.add_argument("--multiple", help="True if you want to use multiple fragment corruption, False otherwise", action="store_true")
  parser.add_argument("--corruptTrain", help="True if you want to corrupt training data, False otherwise", action="store_true")
  parser.add_argument("--corruptTest", help="True if you want to corrupt test data, False otherwise", action="store_true")
  return parser.parse_args()

def get_data(base_path, data_file, labels_file):
  # load dict of arrays
  dict_data = load(os.path.join(base_path, data_file))
  # extract the first array
  data = dict_data['arr_0']
  # label for each file, corresponding to the class
  labels = pd.read_csv(os.path.join(base_path, labels_file))
  labels = labels['0']

  return data, labels

def get_logo_groups(data):
  groups = np.empty(shape=data.shape[0], dtype=int)
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

  return groups

"""# Data corruption"""

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

# multiple fragment data corruption
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

def get_model():

  DURATION = 4
  SAMPLE_RATE = 8000
  # 32000x1 means greyscale image
  input_shape = (DURATION*SAMPLE_RATE, 1)
  model = models.Sequential()
  # 32 filters, each of size 8x8
  model.add(layers.Conv1D(64, 8, strides=4, activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l2(l=0.001))) 
  #model.add(BatchNormalization())
  model.add(layers.MaxPooling1D(pool_size=4, padding="same")) 
  model.add(layers.Conv1D(64, 8, strides=4, activation="relu", kernel_regularizer=regularizers.l2(l=0.001))) 
  #model.add(BatchNormalization())
  model.add(layers.MaxPooling1D(pool_size=4, padding="same"))
  model.add(layers.Conv1D(128, 4, strides=4, activation="relu", kernel_regularizer=regularizers.l2(l=0.001)))
  #model.add(BatchNormalization())
  model.add(layers.MaxPooling1D(pool_size=4, padding="same"))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l=0.001))) 
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(64,activation="relu", kernel_regularizer=regularizers.l2(l=0.001)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(10, activation="softmax"))

  #model.summary()

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

# YAMNet: range (-1,1); 16 kHz sr; Mono-channel
def get_embeddings_yamnet(training_set, test_set): 
  print("Extracting embeddings..")
  yamnet = hub.load("https://tfhub.dev/google/yamnet/1") 
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

def get_classifier():
    classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8192), dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)), 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)), 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)), 
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax") 
    ], name='classifier')

    classifier.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam",
                 metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=2, 
                                            restore_best_weights=True)
    return classifier

def run_model(dataset, labels, corrupt_train, corrupt_test, corruption_length, batch_size=128, epochs=15, multiple=True, pretrained=False):

  if(pretrained):
    dataset = dataset.reshape(-1,64000,1)
  else:
    dataset = dataset.reshape(-1,32000,1)
  logo = LeaveOneGroupOut()
  i=1
  scores = {} #'1': [1.35, 0.61]
  histories = {} 
  predicted_labels = {}
  cm_old = np.zeros((10,10))
  cm = np.zeros((10,10))
  groups = get_logo_groups(dataset)

  for tr_idx, ts_idx in logo.split(dataset, labels, groups):
    print(" (TEST) FOLD: ", i)

    X_tr = dataset[tr_idx, :, :]
    y_tr = labels[tr_idx]
    X_ts = dataset[ts_idx, :, :]
    y_ts = labels[ts_idx]

    if(corrupt_train and corrupt_test):
      mode = "C_all"
      X_tr = corrupt_data(X_tr, corruption_length, multiple_fragments=multiple)
      X_ts = corrupt_data(X_ts, corruption_length, multiple_fragments=multiple)
    elif(corrupt_train):
      mode = "C_tr"
      X_tr = corrupt_data(X_tr, corruption_length, multiple_fragments=multiple)
    elif(corrupt_test):
      mode = "C_ts"
      X_ts = corrupt_data(X_ts, corruption_length, multiple_fragments=multiple)
    else:
      mode = "NC"

    if(pretrained):
      model_name = "yamnet"
      model = get_classifier()
      X_tr, X_ts = get_embeddings_yamnet(X_tr, X_ts)
    else:
      model_name = "cnn"
      model = get_model()

    history = model.fit(X_tr, y_tr, batch_size=batch_size, epochs=epochs)
    # [0] = loss, [1] = accuracy (evaluate)
    score = model.evaluate(X_ts, y_ts)

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

  results = {
  "mode": mode,
  "predicted_labels": predicted_labels,
  "model_name":model_name,
  "confusion_matrix":cm,
  "accuracies": [v[1] for k,v in scores.items()],
  "corr_length": corruption_length,
  "multiple":multiple,
  "scores":scores 
  }

  return results 

def save_results(results):

  metadata = pd.read_csv("./UrbanSound8K/metadata/UrbanSound8K.csv")
  classes=np.unique(metadata["class"])
  mapping=classes.tolist() 

  df = pd.DataFrame()
  for pred in results["predicted_labels"].values():
    pred=pred.reshape(1, -1)
  df = df.transpose()
  df = df.append(pd.DataFrame(results["predicted_labels"].values()))
  df.fillna(-1, inplace=True)

  if(results["multiple"]):
    multiple_format = "MF"
  else:
    multiple_format = "SF"

  run = pd.DataFrame(np.array(results["accuracies"]).reshape(1,10), columns=results["scores"].keys())

  disp = ConfusionMatrixDisplay(confusion_matrix=results["confusion_matrix"], display_labels=mapping)
  fig1, ax = plt.subplots(figsize=(8,8))
  disp.plot(xticks_rotation='vertical', ax=ax)
  
  fig2, ax = plt.subplots(figsize=(8,4))
  plt.bar(results["scores"].keys(), results["accuracies"])
  plt.title("Test accuracy logo cross validation")
  plt.xticks(np.arange(1,11, 1))
  plt.xlabel('Test folder')
  plt.ylabel('Accuracy')
  plt.show()

  if(results["mode"] == 'NC'):
    try:
      run.to_csv("./UrbanSound8K/plots/{}/{}/accuracies.csv".format(results["model_name"],results["mode"]), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}/predictions.csv".format(results["model_name"],results["mode"]), index=False)
      fig1.savefig("./UrbanSound8K/plots/{}/{}/confusionmatrix.png".format(results["model_name"],results["mode"]), format="png") 
      fig2.savefig("./UrbanSound8K/plots/{}/{}/logoaccuracies.png".format(results["model_name"],results["mode"]), format="png") 
    # FileNotFoundError is a subclass of OSError 
    except OSError:
      os.makedirs("./UrbanSound8K/plots/{}/{}".format(results["model_name"],results["mode"]))
      run.to_csv("./UrbanSound8K/plots/{}/{}/accuracies.csv".format(results["model_name"],results["mode"]), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}/predictions.csv".format(results["model_name"],results["mode"]), index=False) 
      fig1.savefig("./UrbanSound8K/plots/{}/{}/confusionmatrix.png".format(results["model_name"],results["mode"]), format="png") 
      fig2.savefig("./UrbanSound8K/plots/{}/{}/logoaccuracies.png".format(results["model_name"],results["mode"]), format="png")   
  else:        
    try:
      run.to_csv("./UrbanSound8K/plots/{}/{}/sim-{}-{}/accuracies.csv".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}/sim-{}-{}/predictions.csv".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), index=False)
      fig1.savefig("./UrbanSound8K/plots/{}/{}/sim-{}-{}/confusionmatrix.png".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), bbox_inches='tight', transparent=True)
      fig2.savefig("./UrbanSound8K/plots/{}/{}/sim-{}-{}/logoaccuracies.png".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), bbox_inches='tight', transparent=True)
    except OSError:
      os.makedirs("./UrbanSound8K/plots/{}/{}/sim-{}-{}".format(results["model_name"],results["mode"],results["corr_length"], multiple_format))
      run.to_csv("./UrbanSound8K/plots/{}/{}/sim-{}-{}/accuracies.csv".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}/sim-{}-{}/predictions.csv".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), index=False)
      fig1.savefig("./UrbanSound8K/plots/{}/{}/sim-{}-{}/confusionmatrix.png".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), bbox_inches='tight', transparent=True)
      fig2.savefig("./UrbanSound8K/plots/{}/{}/sim-{}-{}/logoaccuracies.png".format(results["model_name"],results["mode"],results["corr_length"], multiple_format), bbox_inches='tight', transparent=True)