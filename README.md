## Sound classification in the presence of missing samples

The leading objective of this work is to recognize sound content in the presence of missing signal parts. It falls in the field of Computational Auditory Scene Analysis, CASA. This field is currently focusing its attention on analyzing realistic scenarios in which it is ordinary to find particular conditions such as non-stationary recording circumstances, reverbrant effects or general audio deteriorations. In fact, it is usual to find accidental distortions that influence the signal quality. Another kind of distortion could be represented by a packet-loss in a VoIP (voice-over-IP) transmission. These conditions are generally called corruption and they are aimed at simulating real adverse conditions. Hence, they are artificially reproduced. The main step consists of predicting those corrupted audio files. In order to fulfill this purpose, an audio classification has been performed using different models and approaches with dissimilar peculiarities. Then, the audio reconstruction problem has been faced to provide coherent information in place of the missing parts. 

## Summary:
* Classification with both CNN and (pretrained) YAMNet, working with 1D inputs (raw waveforms; vectors of length 32000 and 64000 respectively).
[Experiments with: no corruption, only training set corruption, only test set corruption, train and test sets corruption (with corruption lengths of 250/32000, 500, 1000, 2000, 4000, 8000)]
* Hyperparameter tuning in order to obtain better classification results 
* Reconstruction using a GAN, passing through the mel spectrogram
* Classification after reconstruction, directly giving YAMNet the reconstructed spectrograms (to be completed) 

The code is structured in the following way:
1. ***data_creation.py***: script that creates npz files containing waveforms (both for the 8000 and 16000 s.rate cases).
2. ***helper_functions.py***: script that contains all the functions to run the CNN model and the pretrained (YAMNet) model. Moreover, in this script we can find the functions aimed at corrupting the audio files and those dedicated to the results saving. 
3. ***models_run.py***: actual script dedicated to the execution of the above mentioned models. 
4. ***hp_tuning.py*** and ***hp_tuning_yamnet.py*** : hyperparameter tuning of the models.
5. ***gan_16khz.py***: Generative Adversarial Network is run in this script in order to obtain a reconstructed version of the dataset, passing through the mel-spectrogram and getting it ready for YAMNet. 
6. ***embeddings_extraction.py***: embeddings extraction script, aimed at putting the data in the exact format for being classified. 
7. ***yamnet_embeddings_classification.py***: script that lastly classify the reconstructed data. 

## List of commands: 
**Classification**
python models_run.py 
- --pretrained (if you want to use YAMNet, otherwise it automatically select the CNN) 
- --multiple (if you want to use multiple fragments corruption, otherwise it automatically select the single-fragment one)
- --corruptTrain (if you want to corrupt the training data)
- --corruptTest (if you want to corrupt the test data)
- --corruptionSize (to be followed by the corruption size e.g. 500, 1000, 2000, etc..)

**Reconstruction**
python gan_16khz.py
- --corruptionSize (to be followed by the corruption size you want to reconstruct with the GAN)

**Classification after reconstruction**
python embeddings_extraction.py & python yamnet_embeddings_classification.py 
- --corruptionSize (to be followed by the corruption size for which you want to extract embeddings)
- --corruptionSize (to be followed by the reconstructed portion size to perform classification)
