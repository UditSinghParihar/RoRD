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
        # self.dataset_H = []
        self.valid_images = []
        self.cropSize=cropSize

    def getImageFiles(self):
        img_files = []
        img_path = "dense/images"
        if self.train:        
            with open(os.path.join(self.base_path, "train_scenes.txt.bkp")) as f:
                scenes = f.read().strip("\n").split("\n")
        else:
            with open(os.path.join(self.base_path, "valid_scenes.txt")) as f:
                scenes = f.read().strip("\n").split("\n")
        print("[INFO]",scenes)
        for scene in scenes:
            image_dir = os.path.join(self.base_path, scene, img_path)
            img_names = os.listdir(image_dir)
            img_files += [os.path.join(image_dir, img) for img in img_names]
        return img_files

    def imgCrop(self, img1):
        # print(img1.size)
        w, h = img1.size
        left = np.random.randint(low = 0, high = w - (self.cropSize))
        upper = np.random.randint(low = 0, high = h - (self.cropSize))

        cropImg = img1.crop((left, upper, left+self.cropSize, upper+self.cropSize))
        
        # cropImg = cv2.cvtColor(np.array(cropImg), cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image", cropImg)
        # cv2.waitKey(0)

        return cropImg

    def getGrid(self, im1, im2, H, scaling_steps=3):

        # im1 = np.array(img1)
        # im2 = np.array(img2)

        # h1, w1 = int(cropSize/(2**scaling_steps)), int(cropSize/(2**scaling_steps))
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
            # if(2 < x < (cropSize-2) and 2 < y < (cropSize-2)):
            # if(20 < x < (im1.shape[0]-20) and 20 < y < (im1.shape[1]-20)):
            if(2 < x < (im1.shape[0]-2) and 2 < y < (im1.shape[1]-2)):
                ids.append(i)
        pos1 = pos1[:, ids]
        pos2 = pos2[:, ids]

        # for i in range(0, pos1.shape[1], 20):
        #   im1 = cv2.circle(im1, (pos1[1, i], pos1[0, i]), 1, (0, 0, 255), 2)
        # for i in range(0, pos2.shape[1], 20):
        #   im2 = cv2.circle(im2, (pos2[1, i], pos2[0, i]), 1, (0, 0, 255), 2)

        # im3 = cv2.hconcat([im1, im2])

        # for i in range(0, pos1.shape[1], 20):
        #   im3 = cv2.line(im3, (int(pos1[1, i]), int(pos1[0, i])), (int(pos2[1, i]) +  im1.shape[1], int(pos2[0, i])), (0, 255, 0), 1)

        # cv2.imshow('Image', im1)
        # cv2.imshow('Image2', im2)
        # cv2.imshow('Image3', im3)
        # cv2.waitKey(0)

        return pos1, pos2
    
    def imgRotH(self, img1, min=0, max=360):
        # im1 = np.array(img1)
        # print(im1.shape, img1.size)
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

        # img2 = cv2.warpPerspective(im1, H, dsize=(width,height))

        #cv2.imshow("Image", img2)
        #cv2.waitKey(0)

        return H, theta

    def build_dataset(self):
        # if self.train:
        #     cache_path = os.path.join(self.base_path, "ipr_PT_2.gz")

        #     if os.path.exists(cache_path):
        #         self.dataset_H, self.valid_images = joblib.load(cache_path)
        #         return

        print("Building Dataset.")

        imgFiles = self.getImageFiles()

        for idx in tqdm(range(len(imgFiles))):

            img = imgFiles[idx]
            img1 = Image.open(img)

            if(img1.mode != 'RGB'):
                img1 = img1.convert('RGB')
            if(img1.size[0] < self.cropSize or img1.size[1] < self.cropSize):
                continue

            # H, theta = self.imgRotH(img1, min=0, max=360)
            # self.dataset_H.append((H, theta))
            self.valid_images.append(img)

            # self.dataset.append((img1, img2, pos1, pos2, H))
            # if len(self.dataset)>10: break
        # self.dataset
        # if self.train:
        #     joblib.dump((self.dataset_H, self.valid_images), cache_path, 3)

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
                print("big fucked")
                exit
            except:
                del self.valid_images[idx]
                # del self.dataset_H[idx]    


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

