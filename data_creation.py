# -*- coding: utf-8 -*-

# importing libraries
import librosa
from glob import glob
import numpy as np
import librosa.display
import tensorflow.keras as keras
import pandas as pd
from numpy import savez_compressed

# path to the different folders
fold1 = glob(r"..\UrbanSound8K\audio\fold1\*.wav")
fold2 = glob(r"..\UrbanSound8K\audio\fold2\*.wav")
fold3 = glob(r"..\UrbanSound8K\audio\fold3\*.wav")
fold4 = glob(r"..\UrbanSound8K\audio\fold4\*.wav")
fold6 = glob(r"..\UrbanSound8K\audio\fold6\*.wav")
fold5 = glob(r"..\UrbanSound8K\audio\fold5\*.wav")
fold7 = glob(r"..\UrbanSound8K\audio\fold7\*.wav")
fold8 = glob(r"..\UrbanSound8K\audio\fold8\*.wav")
fold9 = glob(r"..\UrbanSound8K\audio\fold9\*.wav")
fold10 = glob(r"..\UrbanSound8K\audio\fold10\*.wav")

# datasets for training and testing
dataset = fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8 + fold9 +fold10

# The sample rate is the number of samples per second in a sound
SAMPLE_RATE = 8000
DURATION = 4

audio_features = []
for file in dataset: 
    # sr = None means native sr (44kHz)
    sound_file, sample_rate = librosa.load(file, sr=SAMPLE_RATE)
    # from (n,) to (n,1)
    sound_file = sound_file.reshape(-1,1)
    # normalize mean 0, variance 1
    sound_file = (sound_file - np.mean(sound_file)) / np.std(sound_file)
    audio_length = len(sound_file)
    
    if audio_length < SAMPLE_RATE*DURATION:             
        sound_file = np.concatenate((sound_file, np.zeros(shape=(SAMPLE_RATE*DURATION - audio_length, 1))))
        print('Pad New length =', len(sound_file))
    elif audio_length > SAMPLE_RATE*DURATION:
        sound_file = sound_file[0:SAMPLE_RATE*DURATION]
        print('Cut New length =', len(sound_file))  

    # from (n,1) to (n,); "reshape(-1)" flattens the array
    sound_file = sound_file.reshape(-1)
    audio_features.append(sound_file) 

savez_compressed('dataset8kHz.npz', audio_features)
# label for each file, corresponding to the class
labels = np.array([int((x.split("\\")[-1]).split('-')[1]) for x in dataset]).reshape(-1,1)

y = pd.DataFrame(data=labels)
y.to_csv("./labels.csv")
