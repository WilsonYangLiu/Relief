"""
    Unit tests for ReliefF.
"""

from ReliefF import ReliefF
import csv, os
import numpy as np

if __name__ == '__main__':
	os.chdir(r'E:/Project_ML/ReliefF')
	
	with open(r'data/liver.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile)
		liver = np.array(list(spamreader))
		data = (liver[1:, :-2]).astype(np.float64)
		labels = (liver[1:, -2]).astype(np.int8)
		
	fs = ReliefF(n_features_to_keep=10)
	fs.fit(data, labels)
	data_trans = fs.transform(data)
	
	