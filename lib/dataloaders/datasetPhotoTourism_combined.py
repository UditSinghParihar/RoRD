
import os
import time
import random

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lib.utils import preprocess_image
from lib.utils import preprocess_image, grid_positions, upscale_positions
from lib.dataloaders.datasetPhotoTourism_ipr import PhotoTourismIPR
from lib.dataloaders.datasetPhotoTourism_real import PhotoTourism

from sys import exit, argv
import cv2
import csv

np.random.seed(0)


class PhotoTourismCombined(Dataset):
    def __init__(self, base_path, preprocessing, ipr_pref=0.5, train=True, cropSize=256):
        self.base_path = base_path
        self.preprocessing = preprocessing
        self.cropSize=cropSize

        self.ipr_pref = ipr_pref

        # self.dataset_len = 0
        # self.dataset_len2 = 0

        print("[INFO] Building Original Dataset")
        self.PTReal = PhotoTourism(base_path, preprocessing=preprocessing, train=train, image_size=cropSize)
        self.PTReal.build_dataset()

        # self.dataset_len1 = len(self.PTReal)
        # print("size 1:",len(self.PTReal))
        # for _ in self.PTReal:
        #     pass
        # print("size 2:",len(self.PTReal))
        self.dataset_len1 = len(self.PTReal)
        # joblib.dump(self.PTReal.dataset, os.path.join(self.base_path, "orig_PT_2.gz"), 3)

        print("[INFO] Building IPR Dataset")
        self.PTipr = PhotoTourismIPR(base_path, preprocessing=preprocessing, train=train, cropSize=cropSize)
        self.PTipr.build_dataset()

        # self.dataset_len2 = len(self.PTipr)
        # print("size 1:",len(self.PTipr))
        # for _ in self.PTipr:
        #     pass
        # print("size 2:",len(self.PTipr))
        self.dataset_len2 = len(self.PTipr)

        # joblib.dump((self.PTipr.dataset_H, self.PTipr.valid_images), os.path.join(self.base_path, "ipr_PT_2.gz"), 3)

    def __getitem__(self, idx):
        if random.random()<self.ipr_pref:
            return (self.PTipr[idx%self.dataset_len1], 1)
        return (self.PTReal[idx%self.dataset_len2], 0)

    def __len__(self):
        return self.dataset_len2+self.dataset_len1


if __name__=="__main__":
    pt = PhotoTourismCombined("/scratch/udit/phototourism/", 'caffe', 256)
    dl = DataLoader(pt, batch_size=1, num_workers=2)
    for _ in dl:
        pass
