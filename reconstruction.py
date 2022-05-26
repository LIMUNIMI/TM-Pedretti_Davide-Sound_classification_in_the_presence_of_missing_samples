from __future__ import print_function, division
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
from keras.layers import Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# load dict of arrays
dict_data = load("./UrbanSound8K/dati/dataset8kHz.npz")
# extract the first array
data = dict_data['arr_0']
# label for each file, corresponding to the class
labels = pd.read_csv("./UrbanSound8K/dati/all_labels.csv")
labels = labels['0']
metadata = pd.read_csv("./UrbanSound8K/metadata/UrbanSound8K.csv")
classes=np.unique(metadata["class"])
mapping=classes.tolist() 

data = data.reshape(-1,32000,1)

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

class ContextEncoder():
    def __init__(self):
        # immagini di 128x63 (melspectrogram) e corruzione di 128x16 (corrispondente ad una corruzione di 8000 sulla waveform); numero di canali: 1 
        self.img_rows = 128
        self.img_cols = 63
        self.mask_height = 128
        self.mask_width = 16
        self.channels = 1
        self.num_classes = 2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.channels)
        self.sr=8000

        #optimizer = Adam(0.02, 0.5)
        optimizer = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        gen_missing = self.generator(masked_img)
        print(gen_missing.shape)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_img , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # Encoder
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        model.add(UpSampling2D(size=(4,2)))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(4,2)))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation('tanh'))

        model.summary()

        masked_img = Input(shape=self.img_shape) #immagine a cui manca un pezzo
        gen_missing = model(masked_img) # pezzo ricostruito 

        return Model(masked_img, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.missing_shape) # pezzo mancante ricostruito 
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs_original):
        imgs = copy.deepcopy(imgs_original)
        
        # corrompo la waveform 
        corruption_length=8000
        masked_imgs = []
        missing_parts = np.empty((imgs.shape[0], corruption_length, self.channels))
        masked_imgs_mel = []
        missing_parts_mel = []
        x1 = []
        x2 = []
        for i, sample in enumerate(imgs):
          rand1 = random.randint(0, int(len(sample)/1.3)) #24000 
          zeros = np.zeros(shape=sample[rand1:].shape)
          if (sample[rand1:]==zeros).all() == True or sample[rand1:rand1+corruption_length, :].shape[0]<8000:
              rand1 = random.randint(0, int(len(sample)/6)) # 5000 
              missing_parts[i]=sample[rand1:rand1+corruption_length, :]
          else:
              missing_parts[i]=sample[rand1:rand1+corruption_length, :]
          sample[rand1:rand1+corruption_length] = 0 
          masked_imgs.append(sample)
          # converto la waveform in mel spectrogram
          masked_imgs_mel.append(librosa.feature.melspectrogram(y=sample.reshape(-1), sr=self.sr))
          # converto il frammento corrotto in mel spectrogram
          missing_parts_mel.append(librosa.feature.melspectrogram(y=missing_parts[i].reshape(-1), sr=self.sr))
          x1.append(rand1)
          x2.append(rand1+corruption_length)

        return np.array(masked_imgs), missing_parts, np.array(masked_imgs_mel), np.array(missing_parts_mel), np.array(x1), np.array(x2)

    def train(self, epochs, batch_size=128, sample_interval=50):

        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels.values, test_size=0.2, shuffle=False)
        self.X_train = data 
        self.y_train = labels.values

        # Extract classes (at the beginning it was only two classes)
        X0 = self.X_train[(self.y_train == 0).flatten()]
        X1 = self.X_train[(self.y_train == 1).flatten()]
        X2 = self.X_train[(self.y_train == 2).flatten()]
        X3 = self.X_train[(self.y_train == 3).flatten()]
        X4 = self.X_train[(self.y_train == 4).flatten()]
        X5 = self.X_train[(self.y_train == 5).flatten()]
        X6 = self.X_train[(self.y_train == 6).flatten()]
        X7 = self.X_train[(self.y_train == 7).flatten()]
        X8 = self.X_train[(self.y_train == 8).flatten()]
        X9 = self.X_train[(self.y_train == 9).flatten()]

        # Stack arrays in sequence vertically (row wise)
        self.X_train = np.vstack((X0, X1, X2, X3, X4, X5, X6, X7, X8, X9))

        from sklearn.preprocessing import MinMaxScaler
        self.mms = MinMaxScaler(feature_range=(-1,1))
        from sklearn.preprocessing import MaxAbsScaler
        self.mas = MaxAbsScaler()

        # Rescale -1 to 1
        self.X_train = self.mms.fit_transform(self.X_train.reshape(self.X_train.shape[0], 32000))
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 32000, 1)
        #y_train = y_train.reshape(-1, 1)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
          # ---------------------
          #  Train Discriminator
          # ---------------------

          # Select a random batch of images
          idx = np.random.randint(0, self.X_train.shape[0], batch_size)
          imgs = self.X_train[idx]

          # both referred to mel
          _, _, masked_imgs, missing_parts, _, _= self.mask_randomly(imgs)
          #print("Missing parts shape: ", missing_parts.shape, np.max(missing_parts[0]), np.min(missing_parts[0]))
         
          # masked_imgs.shape = (64, 128, 63) dimensione mel spectrogram associato a waveform 32000,1 (batch_size, row, cols)
          # missing_parts.shape = (64, 128, 16) dimensione mel spetrogram associato a frammento corroto 8000,1 (batch_size, row, cols)
          
          #masked_imgs, missing_parts = self.resize_and_reshape(masked_imgs, missing_parts, (128, 63, 1), (128, 16, 1))

          # Generate a batch of new images
          gen_missing = self.generator.predict(masked_imgs)
          original_shape = gen_missing.shape
          gen_missing = self.mas.fit_transform(gen_missing.reshape(gen_missing.shape[0], -1)).reshape(original_shape)
          #librosa.display.waveplot(librosa.feature.inverse.mel_to_audio(missing_parts[1]))
          #plt.title("Corrupted part fed to the net ")
          #plt.show()

          # Train the discriminator          
          d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
          d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
          d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

          # ---------------------
          #  Train Generator
          # ---------------------

          g_loss = self.combined.train_on_batch(masked_imgs, [missing_parts, valid])

          # Plot the progress
          print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
          
          # If at save interval => save generated image samples
          if epoch % sample_interval == 0:
              idx = np.random.randint(0, self.X_train.shape[0], 6)
              imgs = self.X_train[idx]
              #self.sample_images(epoch, imgs)
    
    # bozza per salvare dataset ricostruito
    def save_images(self, imgs, path):
       reconstructed_audio = []
       from tqdm import tqdm # progress bar
       masked_imgs, missing_parts, masked_imgs_mel, missing_parts_mel, x1, x2 = self.mask_randomly(imgs)
       gen_missing = self.generator.predict(masked_imgs_mel)
       '''
       original_shape = gen_missing.shape
       gen_missing = self.mas.fit_transform(gen_missing.reshape(gen_missing.shape[0], -1)).reshape(original_shape)
       '''
       print("Shape save images: ",gen_missing[:].shape) # (8732, 128, 16, 1)
       cnt = 0 #contatore
       for i in tqdm(range(int(imgs.shape[0]))): # 8732
         try:
          tmp = gen_missing[i].reshape(128,16) # tmp.dtype = float32 
          mel_to_wave_missing = librosa.feature.inverse.mel_to_audio(tmp, sr=self.sr)
          # mel to wave missing shape: (7680,)
          mel_to_wave_missing = self.mas.fit_transform(mel_to_wave_missing.reshape(-1, 1))
          zeros = np.zeros((8000,1)) 
          zeros[:mel_to_wave_missing.shape[0], :] = mel_to_wave_missing
          recon_wave = copy.deepcopy(masked_imgs[i, :, :])
          recon_wave[x1[i]:x2[i], :] = zeros
          reconstructed_audio.append(recon_wave)
         except ParameterError:
           cnt+=1
           continue         
      
       savez_compressed(path, reconstructed_audio)

    def sample_images(self, epoch, imgs):
        r, c = 3, 6        

        masked_imgs, missing_parts, masked_imgs_mel, missing_parts_mel, x1, x2 = self.mask_randomly(imgs) #x1.shape = (#images, 1)

        # Prima del resize and reshape: masked_imgs_mel.shape (6, 128, 63), missing_parts_mel.shape (6, 128, 16)
        # masked_imgs_mel, missing_parts_mel = self.resize_and_reshape(masked_imgs_mel, missing_parts_mel, (128, 63, 1), (128, 16, 1))
        # Dopo il resize and reshape: masked_imgs_mel.shape (6, 128, 63, 1), missing_parts_mel.shape (6, 128, 16, 1)
       
        # parte ricostruita del mel spectrogram (8x8)
        gen_missing = self.generator.predict(masked_imgs_mel)
        print(gen_missing.dtype) # float32 
        print(gen_missing[1].dtype) # float32        
        
        # normalizzare il valore gen_missing dividendo per il massimo del valore assoluto 
        print("GEN MISSING SHAPE: ", gen_missing.shape)

        '''
        imgs = 0.5 * imgs + 0.5
        masked_imgs = 0.5 * masked_imgs + 0.5
        gen_missing = 0.5 * gen_missing + 0.5
        '''

        for i in range(3):
  
          # imgs.shape:  (6, 32000, 1)
          # imgs[i, :, :].shape:  (32000, 1)
          #waveform originale
          librosa.display.waveplot(imgs[i, :, :].reshape(-1), sr=8000)
          plt.title("Original")
          plt.show()

          #waveform corrotta
          librosa.display.waveplot(masked_imgs[i, :, :].reshape(-1), sr=8000)
          plt.title("Corrupted")
          plt.show()

          tmp = gen_missing[i].reshape(128,16)
                           
          #print("Max value reconstructed piece: ", np.max(tmp))
          #print("Min value reconstructed piece: ", np.min(tmp))         
          #print("Gen missing reshape, expected (128,16)", tmp.shape)
          mel_to_wave_missing = librosa.feature.inverse.mel_to_audio(tmp, sr=self.sr)
          mel_to_wave_missing = self.mas.fit_transform(mel_to_wave_missing.reshape(-1, 1)) 
          #print("Recon wave shape, expected (circa 8000): ", mel_to_wave_missing.shape) # (circa 8000,1)
          zeros = np.zeros((8000,1)) 
          zeros[:mel_to_wave_missing.shape[0], :] = mel_to_wave_missing    
          # mel to wave con padding per colmare gli 8000     
          #print("Corrupted fragment shape wave, expected (8000, ):", zeros.shape) # (8000,)
          recon_wave = copy.deepcopy(masked_imgs[i, :, :])
          recon_wave[x1[i]:x2[i], :] = zeros
          #print("Recon wave shape, expected (32000,1)", recon_wave.shape)
          # wave ricostruita (32000, 1)
          # recon_wave = self.mas.fit_transform(recon_wave) 
          librosa.display.waveplot(recon_wave.reshape(-1), sr=self.sr)
          plt.title("Reconstrucred waveform")
          plt.show()
          #reconstructed = masked_imgs[i, :, :]
          print("")
          print("*"*50)
          '''
          librosa.display.waveplot(mel_to_wave_missing.reshape(-1), sr=self.sr)
          plt.title("Missing piece reconstructed")
          plt.show()
          '''

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

if __name__ == '__main__':
    context_encoder = ContextEncoder()
    context_encoder.train(epochs=50, batch_size=64, sample_interval=10)
    context_encoder.save_images(context_encoder.X_train, "./UrbanSound8K/dati/reconstructed_8kHz.npz")