"""
    Unit tests for ReliefF.
"""

from ReliefF import ReliefF
import csv, os
import numpy as np

if __name__ == '__main__':
	#os.chdir(r'E:/Project_ML/ReliefF')

	with open(r'data/liver.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile)
		liver = np.array(list(spamreader))
		data = (liver[1:, :-2]).astype(np.float64)
		labels = (liver[1:, -2]).astype(np.int8)
		
	fs = ReliefF(n_features_to_keep=10)
	fs.fit(data, labels)
	data_trans = fs.transform(data)
	'''
	tmp = np.array([[ 1.61311827,  0.19955703],
		[-0.21997067,  0.86474714],
		[-0.58658846, -1.46341823],
		[-1.31982404, -0.79822813],
		[ 0.5132649 ,  1.19734219]])
	
	y = np.array([1, 1, 1, 2, 2])
	fs = ReliefF(2)
	fs.fit(tmp, y)
	'''