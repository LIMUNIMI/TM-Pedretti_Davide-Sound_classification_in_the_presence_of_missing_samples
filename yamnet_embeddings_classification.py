import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy
import os 
from numpy import savez_compressed, load
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda,InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, Conv2DTranspose
from keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import gc
import IPython.display as ipd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from helper_functions import * 

SEED=42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC']='1'

yamnet = keras.models.load_model("./saved_model/yamnet")
labels = pd.read_csv("./UrbanSound8K/dati/all_labels.csv")
labels = labels['0']

# get classification network 
def get_mel_yamnet_classification():
  mel_yamnet_classification = Sequential() ## Create empty new model
  mel_yamnet_classification.add(InputLayer(input_shape=(8*1024))) ### Create input shape with the right dimension
  mel_yamnet_classification.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001294791175975606)))
  mel_yamnet_classification.add(tf.keras.layers.Dropout(0.3863790562689953)) 
  mel_yamnet_classification.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001294791175975606)))
  mel_yamnet_classification.add(tf.keras.layers.Dropout(0.3863790562689953))
  mel_yamnet_classification.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001294791175975606)))
  mel_yamnet_classification.add(tf.keras.layers.Dropout(0.3863790562689953))
  mel_yamnet_classification.add(tf.keras.layers.Dense(10, activation="softmax"))
  #mel_yamnet_classification.add(Dense(10,activation="softmax"))
  return mel_yamnet_classification

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

def run_final_model(labels, corruption_length, batch_size=128, epochs=30):

  if(corruption_length==500):
    print("***YAMNet run with reconstructed dataset (initial corruption of 500)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_500.npz")
  elif(corruption_length==1000):
    print("***YAMNet run with reconstructed dataset (initial corruption of 1000)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_1000.npz")
  elif(corruption_length==2000):
    print("***YAMNet run with reconstructed dataset (initial corruption of 2000)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_2000.npz")
  elif(corruption_length==4000):
    print("***YAMNet run with reconstructed dataset (initial corruption of 4000)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_4000.npz")
  elif(corruption_length==8000):
    print("***YAMNet run with reconstructed dataset (initial corruption of 8000)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_8000.npz")
  elif(corruption_length==16000):
    print("***YAMNet run with reconstructed dataset (initial corruption of 16000)***")
    dataset_rec = load("./UrbanSound8K/dati/embeddings_16000.npz")

  dataset = dataset_rec['arr_0']
  model_name = "yamnet_after_reconstruction"
  mode = "reconstructed"
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

    X_tr = dataset[tr_idx, :]
    y_tr = labels[tr_idx]
    X_ts = dataset[ts_idx, :]
    y_ts = labels[ts_idx]
    
    model = get_mel_yamnet_classification()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 optimizer="adam", # default learning rate
                 metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=2, 
                                            restore_best_weights=True) 
                                       
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
  "model_name":model_name,
  "predicted_labels": predicted_labels,
  "corr_length": corruption_length,
  "confusion_matrix":cm,
  "accuracies": [v[1] for k,v in scores.items()],
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

  if(results["mode"] == 'reconstructed'):
    try:
      run.to_csv("./UrbanSound8K/plots/{}/{}-{}/accuracies.csv".format(results["model_name"],results["mode"], results["corr_length"]), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}-{}/predictions.csv".format(results["model_name"],results["mode"], results["corr_length"]), index=False)
      fig1.savefig("./UrbanSound8K/plots/{}/{}-{}/confusionmatrix.svg".format(results["model_name"],results["mode"],results["corr_length"]), format="svg") 
      fig2.savefig("./UrbanSound8K/plots/{}/{}-{}/logoaccuracies.svg".format(results["model_name"],results["mode"], results["corr_length"]), format="svg") 
    # FileNotFoundError is a subclass of OSError 
    except OSError:
      os.makedirs("./UrbanSound8K/plots/{}/{}-{}".format(results["model_name"],results["mode"], results["corr_length"]))
      run.to_csv("./UrbanSound8K/plots/{}/{}-{}/accuracies.csv".format(results["model_name"],results["mode"], results["corr_length"]), index=False)
      df.to_csv("./UrbanSound8K/plots/{}/{}-{}/predictions.csv".format(results["model_name"],results["mode"], results["corr_length"]), index=False) 
      fig1.savefig("./UrbanSound8K/plots/{}/{}-{}/confusionmatrix.svg".format(results["model_name"],results["mode"], results["corr_length"]), format="svg") 
      fig2.savefig("./UrbanSound8K/plots/{}/{}-{}/logoaccuracies.svg".format(results["model_name"],results["mode"], results["corr_length"]), format="svg")   

############################### Saving the results ###################################

def read_files(files, root_dir_path, mean_accuracy, mean_std, sim_name, mode, corruption_length):  
  for f in files:
    if(f.endswith("es.csv")):      
      dfr = pd.read_csv(root_dir_path+'/'+ f)

      accuracy = np.mean(dfr.values)
      try:
        mean_accuracy[mode+"_"+corruption_length].append(accuracy)
      except KeyError:
        mean_accuracy[mode+"_"+corruption_length]=[accuracy]

      std = np.std(dfr.values)
      try:
        mean_std[mode+"_"+corruption_length].append(std)
      except KeyError:
        mean_std[mode+"_"+corruption_length]=[std]

      try:
        sim_name[mode+"_"+corruption_length].append(root_dir_path.rpartition('/')[-1])
      except KeyError:
        sim_name[mode+"_"+corruption_length]= [root_dir_path.rpartition('/')[-1]]

  return sim_name, mean_accuracy

# function to load the results from the different folders
def load_results(path):
  mean_accuracy = {}
  mean_std = {}
  sim_name = {}
  # it takes "./UrbanSound8K" instead of "./UrbanSound8K/dati"
  path = os.path.dirname(path)
  accuracies_df = {}
  for root_dir_path, sub_dirs, files in os.walk(os.path.join(path,'plots')):
    if root_dir_path.__contains__('yamnet'):  
      if("NC" in root_dir_path):
        mode = "no_corruption"
        corruption_length= ""
        sim_name, mean_accuracy = read_files(files, root_dir_path, mean_accuracy, mean_std, sim_name, mode, corruption_length) 
      elif("C_all" in root_dir_path) and root_dir_path.__contains__('SF'):
        mode = "all_corrupted"
        corruption_length=root_dir_path.rpartition('-')[-3].rpartition('/')[-1].rpartition('-')[-1]
        sim_name, mean_accuracy = read_files(files, root_dir_path, mean_accuracy, mean_std, sim_name, mode, corruption_length) 
      elif("reconstructed" in root_dir_path):
        mode = "reconstructed"  
        corruption_length=root_dir_path.rpartition('-')[-1]
        sim_name, mean_accuracy = read_files(files, root_dir_path, mean_accuracy, mean_std, sim_name, mode, corruption_length)        
      if(len(sim_name)):
        try:
          accuracies_df[mode+"_"+corruption_length]=pd.DataFrame({"Simulation name": sim_name[mode+"_"+corruption_length], "Mean accuracy": mean_accuracy[mode+"_"+corruption_length]})
        except KeyError:
          continue
  return accuracies_df

# function to save the overall results
def save_final_results(path):
  results = load_results(path)
  NC = results["no_corruption_"]['Mean accuracy'].values[0]
  data = [[NC, NC, NC, NC, NC],
          [results["all_corrupted_500"]['Mean accuracy'].values[0], 
           results["all_corrupted_1000"]['Mean accuracy'].values[0], 
           results["all_corrupted_2000"]['Mean accuracy'].values[0],
           results["all_corrupted_4000"]['Mean accuracy'].values[0],
           results["all_corrupted_8000"]['Mean accuracy'].values[0]],
          [results["reconstructed_500"]['Mean accuracy'].values[0],
           results["reconstructed_1000"]['Mean accuracy'].values[0],
           results["reconstructed_2000"]['Mean accuracy'].values[0], 
           results["reconstructed_4000"]['Mean accuracy'].values[0],
           results["reconstructed_8000"]['Mean accuracy'].values[0]]]
  X = np.arange(5)
  fig = plt.figure(figsize=(15,6))
  ax = fig.add_axes([0,0,1,1])
  ax.set_ylabel('Accuracy')
  ax.set_xlabel('Corruption length')
  ax.bar(X + 0.00, data[0], color = '#BC8F8F', width = 0.25, edgecolor = 'black')
  ax.bar(X + 0.25, data[1], color = '#3CB371', width = 0.25, edgecolor = 'black')
  ax.bar(X + 0.50, data[2], color = '#87CEEB', width = 0.25, edgecolor = 'black')
  ax.set_xticks(X+0.25)
  ax.set_xticklabels(["500","1000","2000","4000", "8000"]) 
  ax.legend(labels=['No corruption', 'Corruption', 'Reconstruction'])
  try:
    fig.savefig("./UrbanSound8K/plots/overall_results_after_reconstruction/results.svg",bbox_inches='tight', format="svg")
    print("Saved to: {}".format("./UrbanSound8K/plots/overall_results_after_reconstruction/results.svg"))
  except OSError:
    os.makedirs("./UrbanSound8K/plots/overall_results_after_reconstruction")
    fig.savefig("./UrbanSound8K/plots/overall_results_after_reconstruction/results.svg",bbox_inches='tight', format="svg")

if __name__ == '__main__':
  results = run_final_model(labels, corruption_length=2000)
  save_results(results) # save the results of a single execution 
  save_final_results("./UrbanSound8K/dati") # save the overall results 