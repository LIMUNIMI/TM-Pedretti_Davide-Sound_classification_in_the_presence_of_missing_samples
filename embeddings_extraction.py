import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import random 
import copy
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
import librosa
import librosa.display
import gc
import IPython.display as ipd
from tqdm import tqdm
from helper_functions import * 

args = get_args()

SEED=42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC']='1'

yamnet = keras.models.load_model("./saved_model/yamnet")

def frame_spectrograms(reconstructed_spectrograms):
  framed_features=[]
  for x_mel_log_librosa in reconstructed_spectrograms:
    x_mel_log_librosa = np.log(x_mel_log_librosa + 0.001)
    x_mel_log_librosa = np.nan_to_num(x_mel_log_librosa) # replace nan with zeros
    x_mel_log_librosa_framed = np.ndarray((8,96,64))
    j=0
    # Frame size = 96
    # Split the spectrogram into frames, and put the frames in the 3D array
    # step of 48
    for i in range(0,x_mel_log_librosa.shape[1]-96,96//2):
        x_mel_log_librosa_framed[j] = x_mel_log_librosa[:,i:(i+96)].T # Spectrogram shape is (64,96) but tensor
        j+=1
    framed_features.append(x_mel_log_librosa_framed)
  return np.array(framed_features)

# get embeddings network
def get_mel_yamnet_embeddings():
  mel_yamnet_embeddings = Sequential() ## Create empty new model
  mel_yamnet_embeddings.add(InputLayer(input_shape=(96,64))) ### Create input shape with the right dimension
  for i,layer in enumerate(yamnet.layers): # Take all the Yamnet layers after the waveform preprocessing one
      if i >= 21 and i <=103: # From the first layer after preprocessing
          mel_yamnet_embeddings.add(layer) # Add them to the new model
  return mel_yamnet_embeddings

def get_all_embeddings_yamnet(dataset, mel_yamnet_embeddings): 
  print("Extracting embeddings..")
  embeddings=[]

  for i in tqdm(dataset, desc = 'Full set embeddings progress bar'):
    embeddings.append(mel_yamnet_embeddings(i).numpy().flatten().reshape((1,8192)))
  embeddings = np.array(embeddings) # (8732, 1, 8192)
  embeddings = embeddings.reshape(-1, 8192) # (8732, 8192)

  # Alternative approach:
  # embeddings = mel_yamnet_embeddings(dataset) -> [8732,8,1024]
  # embeddings = embeddings.reshape(-1, 8192) # (8732, 8192)
  return embeddings

# load the reconstructed spectrograms and framing them
def framing(corruption_length):
  if(corruption_length==500):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 500)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_500.npz")
  elif(corruption_length==1000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 1000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_1000.npz")
  elif(corruption_length==2000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 2000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_2000.npz")
  elif(corruption_length==4000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 4000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_4000.npz")
  elif(corruption_length==8000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 8000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_8000.npz")
  elif(corruption_length==16000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 16000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_16000.npz")
  elif(corruption_length==20000):
    print("***Embeddings extraction with reconstructed dataset (initial corruption of 20000)***")
    dataset_rec = load("./UrbanSound8K/dati/full_reconstructed_16kHz_20000.npz")
  # framing 
  rec = dataset_rec['arr_0']
  dataset = frame_spectrograms(rec)
  return dataset 

if __name__ == '__main__':
  corruption_length=args.corruptionSize
  data = framing(corruption_length)
  mel_yamnet_embeddings = get_mel_yamnet_embeddings()
  final_data = get_all_embeddings_yamnet(data, mel_yamnet_embeddings)
  print("Final shape: ",final_data.shape) # Expected 8732, 8192
  savez_compressed('./UrbanSound8K/dati/embeddings_%s.npz' % corruption_length, final_data)