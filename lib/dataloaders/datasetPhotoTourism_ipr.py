import os
from sys import exit, argv
import csv
import random

import joblib
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from lib.utils import preprocess_image, grid_positions, upscale_positions

np.random.seed(0)


class PhotoTourismIPR(Dataset):
	def __init__(self, base_path, preprocessing, train=True, cropSize=256):
		self.base_path = base_path
		self.train = train
		self.preprocessing = preprocessing
		self.valid_images = []
		self.cropSize=cropSize

	def getImageFiles(self):
		img_files = []
		img_path = "dense/images"
		if self.train:
			print("Inside training!!")

			with open(os.path.join("configs", "train_scenes_small.txt")) as f:
				scenes = f.read().strip("\n").split("\n")

		print("[INFO]",scenes)
		for scene in scenes:
			image_dir = os.path.join(self.base_path, scene, img_path)
			img_names = os.listdir(image_dir)
			img_files += [os.path.join(image_dir, img) for img in img_names]
		return img_files

	def imgCrop(self, img1):
		w, h = img1.size
		left = np.random.randint(low = 0, high = w - (self.cropSize))
		upper = np.random.randint(low = 0, high = h - (self.cropSize))

		cropImg = img1.crop((left, upper, left+self.cropSize, upper+self.cropSize))
		
		return cropImg

	def getGrid(self, im1, im2, H, scaling_steps=3):
		h1, w1 = int(im1.shape[0]/(2**scaling_steps)), int(im1.shape[1]/(2**scaling_steps))
		device = torch.device("cpu")

		fmap_pos1 = grid_positions(h1, w1, device)
		pos1 = upscale_positions(fmap_pos1, scaling_steps=scaling_steps).data.cpu().numpy()

		pos1[[0, 1]] = pos1[[1, 0]]
		
		ones = np.ones((1, pos1.shape[1]))
		pos1Homo = np.vstack((pos1, ones))
		pos2Homo = np.dot(H, pos1Homo)
		pos2Homo = pos2Homo/pos2Homo[2, :]
		pos2 = pos2Homo[0:2, :]

		pos1[[0, 1]] = pos1[[1, 0]]
		pos2[[0, 1]] = pos2[[1, 0]]
		pos1 = pos1.astype(np.float32)
		pos2 = pos2.astype(np.float32)

		ids = []
		for i in range(pos2.shape[1]):
			x, y = pos2[:, i]

			if(2 < x < (im1.shape[0]-2) and 2 < y < (im1.shape[1]-2)):
				ids.append(i)
		pos1 = pos1[:, ids]
		pos2 = pos2[:, ids]

		return pos1, pos2
	
	def imgRotH(self, img1, min=0, max=360):
		width, height = img1.size
		theta = np.random.randint(low=min, high=max) * (np.pi / 180)
		Tx = width / 2
		Ty = height / 2
		sx = random.uniform(-1e-2, 1e-2)
		sy = random.uniform(-1e-2, 1e-2)
		p1 = random.uniform(-1e-4, 1e-4)
		p2 = random.uniform(-1e-4, 1e-4)

		alpha = np.cos(theta)
		beta = np.sin(theta)

		He = np.matrix([[alpha, beta, Tx * (1 - alpha) - Ty * beta], [-beta, alpha, beta * Tx + (1 - alpha) * Ty], [0, 0, 1]])
		Ha = np.matrix([[1, sy, 0], [sx, 1, 0], [0, 0, 1]])
		Hp = np.matrix([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])

		H = He @ Ha @ Hp

		return H, theta

	def build_dataset(self):
		print("Building Dataset.")

		imgFiles = self.getImageFiles()

		for idx in tqdm(range(len(imgFiles))):

			img = imgFiles[idx]
			img1 = Image.open(img)

			if(img1.mode != 'RGB'):
				img1 = img1.convert('RGB')
			if(img1.size[0] < self.cropSize or img1.size[1] < self.cropSize):
				continue

			self.valid_images.append(img)

	def __len__(self):
		return len(self.valid_images)

	def __getitem__(self, idx):
		while 1:
			try:
				img = self.valid_images[idx]    
				   
				img1 = Image.open(img)
				img1 = self.imgCrop(img1)
				width, height = img1.size

				H, theta = self.imgRotH(img1, min=0, max=360)

				img1 = np.array(img1)
				img2 = cv2.warpPerspective(img1, H, dsize=(width,height))
				img2 = np.array(img2)

				pos1, pos2 =  self.getGrid(img1, img2, H)

				assert (len(pos1) != 0 and len(pos2) != 0)
				break
			except IndexError:
				print("IndexError")
				exit(1)
			except:
				del self.valid_images[idx]

		img1 = preprocess_image(img1, preprocessing=self.preprocessing)
		img2 = preprocess_image(img2, preprocessing=self.preprocessing)

		return {
			'image1': torch.from_numpy(img1.astype(np.float32)),
			'image2': torch.from_numpy(img2.astype(np.float32)),
			'pos1': torch.from_numpy(pos1.astype(np.float32)),
			'pos2': torch.from_numpy(pos2.astype(np.float32)),
			'H': np.array(H),
			'theta': np.array([theta])
		}


if __name__ == '__main__':
	rootDir = argv[1]

	training_dataset = PhotoTourismIPR(rootDir, 'caffe')
	training_dataset.build_dataset()

	data = training_dataset[0]
	print(data['image1'].shape, data['image2'].shape, data['pos1'].shape, data['pos2'].shape, len(training_dataset))
