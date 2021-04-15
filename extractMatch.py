import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm
import time
import scipy
import scipy.io
import scipy.misc
import os
import sys

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import pydegensac


parser = argparse.ArgumentParser(description='Feature extraction script')
parser.add_argument('imgs', type=str, nargs=2)
parser.add_argument(
	'--preprocessing', type=str, default='caffe',
	help='image preprocessing (caffe or torch)'
)

parser.add_argument(
	'--model_file', type=str,
	help='path to the full model'
)

parser.add_argument(
	'--no-relu', dest='use_relu', action='store_false',
	help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

parser.add_argument(
	'--sift', dest='use_sift', action='store_true',
	help='Show sift matching as well'
)
parser.set_defaults(use_sift=False)


def extract(image, args, model, device):
	if len(image.shape) == 2:
		image = image[:, :, np.newaxis]
		image = np.repeat(image, 3, -1)

	input_image = preprocess_image(
		image,
		preprocessing=args.preprocessing
	)
	with torch.no_grad():
		keypoints, scores, descriptors = process_multiscale(
			torch.tensor(
				input_image[np.newaxis, :, :, :].astype(np.float32),
				device=device
			),
			model,
			scales=[1]
		)

	keypoints = keypoints[:, [1, 0, 2]]

	feat = {}
	feat['keypoints'] = keypoints
	feat['scores'] = scores
	feat['descriptors'] = descriptors

	return feat


def rordMatching(image1, image2, feat1, feat2, matcher="BF"):
	if(matcher == "BF"):

		t0 = time.time()
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		matches = bf.match(feat1['descriptors'], feat2['descriptors'])
		matches = sorted(matches, key=lambda x:x.distance)
		t1 = time.time()
		print("Time to extract matches: ", t1-t0)

		print("Number of raw matches:", len(matches))

		match1 = [m.queryIdx for m in matches]
		match2 = [m.trainIdx for m in matches]

		keypoints_left = feat1['keypoints'][match1, : 2]
		keypoints_right = feat2['keypoints'][match2, : 2]

		np.random.seed(0)

		t0 = time.time()

		H, inliers = pydegensac.findHomography(keypoints_left, keypoints_right, 10.0, 0.99, 10000)

		t1 = time.time()
		print("Time for ransac: ", t1-t0)

		n_inliers = np.sum(inliers)
		print('Number of inliers: %d.' % n_inliers)

		inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_left[inliers]]
		inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_right[inliers]]
		placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

		draw_params = dict(matchColor = (0,255,0),
		                   singlePointColor = (255,0,0),
		                   # matchesMask = matchesMask,
		                   flags = 0)
		image3 = cv2.drawMatches(image1, inlier_keypoints_left, image2, inlier_keypoints_right, placeholder_matches, None, **draw_params)

		plt.figure(figsize=(20, 20))
		plt.imshow(image3)
		plt.axis('off')
		plt.show()


def siftMatching(img1, img2):
	img1 = np.array(cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB))
	img2 = np.array(cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB))

	# surf = cv2.xfeatures2d.SURF_create(100)
	surf = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = surf.detectAndCompute(img1, None)
	kp2, des2 = surf.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	good = []
	for m, n in matches:
		if m.distance < 0.7*n.distance:
			good.append(m)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

	model, inliers = pydegensac.findHomography(src_pts, dst_pts, 10.0, 0.99, 10000)

	n_inliers = np.sum(inliers)
	print('Number of inliers: %d.' % n_inliers)

	inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
	inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
	placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
	image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

	cv2.imshow('Matches', image3)
	cv2.waitKey(0)

	src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
	dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)

	return src_pts, dst_pts


if __name__ == '__main__':
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	args = parser.parse_args()

	model = D2Net(
		model_file=args.model_file,
		use_relu=args.use_relu,
		use_cuda=use_cuda
	)

	image1 = np.array(Image.open(args.imgs[0]))
	image2 = np.array(Image.open(args.imgs[1]))

	print('--\nRoRD\n--')
	feat1 = extract(image1, args, model, device)
	feat2 = extract(image2, args, model, device)
	print("Features extracted.")

	rordMatching(image1, image2, feat1, feat2, matcher="BF")

	if(args.use_sift):
		print('--\nSIFT\n--')
		siftMatching(image1, image2)
