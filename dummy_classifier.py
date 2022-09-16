import numpy as np
from numpy import load
import pandas as pd 
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
import random 
import os 
import gc 

SEED=42
np.random.seed(SEED)
random.seed(SEED)
os.environ['TF_CUDNN_DETERMINISTIC']='1'

dict_data = load("./UrbanSound8K/dati/dataset16kHz.npz")
dataset = dict_data['arr_0']
dataset = dataset.reshape(-1,64000,1)
labels = pd.read_csv("./UrbanSound8K/dati/all_labels.csv")
labels = labels['0']

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

def run_model(dataset, labels):

	logo = LeaveOneGroupOut()
	i=1
	groups = get_logo_groups(dataset)
	avg_acc = 0

	for tr_idx, ts_idx in logo.split(dataset, labels, groups):
		model = DummyClassifier(random_state=SEED, strategy = 'stratified') 
		print(" (TEST) FOLD: ", i)

		X_tr = dataset[tr_idx, :, :]
		y_tr = labels[tr_idx]
		X_ts = dataset[ts_idx, :, :]
		y_ts = labels[ts_idx]

		model.fit(X_tr, y_tr)
		predictions = model.predict(X_ts) #(873,10) # predicted labels
		correct = np.where(predictions==y_ts)[0].shape[0] # number of correctly predicted samples
		idx_correct = np.where(predictions==y_ts)[0] # indexes of correctly predicted samples
		print(np.unique(predictions[idx_correct]))
		total = y_ts.shape[0] # tot.number of total samples in test set
		print(correct, total)
		score = correct/total # accuracy
		avg_acc+=score # avg accuracy
		print(score)
		i+=1
		del X_tr, X_ts, y_tr, y_ts, model
		gc.collect()
	print("Avg accuracy on all folds: {}".format(avg_acc/10))  

	return avg_acc/10

if __name__ == '__main__':
	run_model(dataset, labels)
