import numpy as np
import re
import os
import argparse


def natural_sort(l): 
	convert = lambda text: int(text) if text.isdigit() else text.lower() 
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	
	return sorted(l, key = alphanum_key)


def angular_distance_np(R_hat, R):
	# measure the angular distance between two rotation matrice
	# R1,R2: [n, 3, 3]
	if R_hat.shape == (3,3):
		R_hat = R_hat[np.newaxis,:]
	if R.shape == (3,3):
		R = R[np.newaxis,:]
	n = R.shape[0]
	trace_idx = [0,4,8]
	trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
	metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0

	return metric


def main():
	parser = argparse.ArgumentParser(description='Rotation and translation metric.')
	parser.add_argument('--trans1', type=str)
	parser.add_argument('--trans2', type=str)

	args = parser.parse_args()

	transFiles1 = natural_sort([file for file in os.listdir(args.trans1) if (file.find("npy") != -1 )])
	transFiles1 = [os.path.join(args.trans1, img) for img in transFiles1]

	transFiles2 = natural_sort([file for file in os.listdir(args.trans2) if (file.find("npy") != -1 )])
	transFiles2 = [os.path.join(args.trans2, img) for img in transFiles2]

	# print(len(transFiles1), transFiles1)
	# print(len(transFiles2), transFiles2)

	for T1_file, T2_file in zip(transFiles1, transFiles2):
		T1 = np.load(T1_file)
		T2 = np.load(T2_file)
		print("Shapes: ", T1.shape, T2.shape)

		for i in range(T1.shape[2]):
			R1 = T1[:3, :3, i]
			R2 = T2[:3, :3, i]
			t1 = T1[:4, -1, i]
			t2 = T2[:4, -1, i]

			R_norm = angular_distance_np(R1.reshape(1,3,3), R2.reshape(1,3,3))[0]

			print("R norm:", R_norm)
			exit(1)


if __name__ == "__main__":
	main()