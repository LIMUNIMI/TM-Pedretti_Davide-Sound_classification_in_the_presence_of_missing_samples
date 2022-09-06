# Sound classification in the presence of missing samples

The leading objective of this work is to recognise sound content in the presence of missing signal parts. It falls in the field of Computational Auditory Scene Analysis, CASA. This field is currently focusing its attention on analyzing realistic scenarios in which it is ordinary to find particular conditions such as non-stationary recording circumstances, reverbrant effects or general audio deteriorations. In fact, it is usual to find accidental distortions that influence the signal quality. Another kind of distortion could be represented by a packet-loss in a VoIP (voice-over-IP) transmission. These conditions are generally called corruption and they are aimed at simulating real adverse conditions. Hence, they are artificially reproduced. The main step consists of predicting those corrupted audio files. In order to fulfill this purpose, an audio classification will be performed using different models and approaches with dissimilar peculiarities.

Summary:
* Classification with both CNN and (pretrained) YAMNet, working with 1D inputs (raw waveforms; vectors of length 32000 and 64000 respectively).
[Experiments with: no corruption, only training set corruption, only test set corruption, train and test sets corruption (with corruption lengths of 250/32000, 500, 1000, 2000, 4000, 8000)]
* Hyperparameter tuning in order to obtain better classification results 
* Reconstruction using a GAN, passing through the mel spectrogram
* Classification after reconstruction, directly giving YAMNet the reconstructed spectrograms (to be completed) 
