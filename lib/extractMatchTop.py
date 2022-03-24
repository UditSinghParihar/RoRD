import argparse
import numpy as np
import imageio
import torch
from tqdm import tqdm
import time
import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import cv2
import matplotlib.pyplot as plt
import os
from sys import exit, argv
from PIL import Image
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import pydegensac


def extractSingle(image, model, device):

	with torch.no_grad():
		keypoints, scores, descriptors = process_multiscale(
			image.to(device).unsqueeze(0),
			model,
			scales=[1]
		)

	keypoints = keypoints[:, [1, 0, 2]]

	feat = {}
	feat['keypoints'] = keypoints
	feat['scores'] = scores
	feat['descriptors'] = descriptors

	return feat


def siftMatching(img1, img2, HFile1, HFile2, device):
	if HFile1 is not None:
		H1 = np.load(HFile1)
		H2 = np.load(HFile2)

	rgbFile1 = img1
	img1 = Image.open(img1)
	
	if(img1.mode != 'RGB'):
		img1 = img1.convert('RGB')
	img1 = np.array(img1)

	if HFile1 is not None:
		img1 = cv2.warpPerspective(img1, H1, dsize=(400,400))

	#### Visualization ####
	# cv2.imshow("Image", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	# cv2.waitKey(0)

	rgbFile2 = img2
	img2 = Image.open(img2)
	
	if(img2.mode != 'RGB'):
		img2 = img2.convert('RGB')
	img2 = np.array(img2)

	if HFile2 is not None:
		img2 = cv2.warpPerspective(img2, H2, dsize=(400,400))

	#### Visualization ####
	# cv2.imshow("Image", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	# cv2.waitKey(0)

	# surf = cv2.xfeatures2d.SURF_create(100) # SURF
	surf = cv2.xfeatures2d.SIFT_create()

	kp1, des1 = surf.detectAndCompute(img1, None)
	kp2, des2 = surf.detectAndCompute(img2, None)

	matches = mnn_matcher(
			torch.from_numpy(des1).float().to(device=device),
			torch.from_numpy(des2).float().to(device=device)
		)

	src_pts = np.float32([ kp1[m[0]].pt for m in matches ]).reshape(-1, 2)
	dst_pts = np.float32([ kp2[m[1]].pt for m in matches ]).reshape(-1, 2)

	if(src_pts.shape[0] < 5 or dst_pts.shape[0] < 5):
		return [], []

	H, inliers = pydegensac.findHomography(src_pts, dst_pts, 8.0, 0.99, 10000)

	n_inliers = np.sum(inliers)

	inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
	inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
	placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

	#### Visualization ####
	image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)
	image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
	# cv2.imshow('Matches', image3)
	# cv2.waitKey()

	src_pts = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
	dst_pts = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
	
	if HFile1 is None:
		return src_pts, dst_pts, image3, image3
	
	orgSrc, orgDst = orgKeypoints(src_pts, dst_pts, H1, H2)
	matchImg = drawOrg(cv2.imread(rgbFile1), cv2.imread(rgbFile2), orgSrc, orgDst)

	return orgSrc, orgDst, matchImg, image3


def orgKeypoints(src_pts, dst_pts, H1, H2):
	ones = np.ones((src_pts.shape[0], 1))

	src_pts = np.hstack((src_pts, ones))
	dst_pts = np.hstack((dst_pts, ones))

	orgSrc = np.linalg.inv(H1) @ src_pts.T
	orgDst = np.linalg.inv(H2) @ dst_pts.T

	orgSrc = orgSrc/orgSrc[2, :]
	orgDst = orgDst/orgDst[2, :]

	orgSrc = np.asarray(orgSrc)[0:2, :]
	orgDst = np.asarray(orgDst)[0:2, :]

	return orgSrc, orgDst


def drawOrg(image1, image2, orgSrc, orgDst):
	img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

	for i in range(orgSrc.shape[1]):
		im1 = cv2.circle(img1, (int(orgSrc[0, i]), int(orgSrc[1, i])), 3, (0, 0, 255), 1)
	for i in range(orgDst.shape[1]):
		im2 = cv2.circle(img2, (int(orgDst[0, i]), int(orgDst[1, i])), 3, (0, 0, 255), 1)

	im4 = cv2.hconcat([im1, im2])
	for i in range(orgSrc.shape[1]):
		im4 = cv2.line(im4, (int(orgSrc[0, i]), int(orgSrc[1, i])), (int(orgDst[0, i]) +  im1.shape[1], int(orgDst[1, i])), (0, 255, 0), 1)
	im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
	# cv2.imshow("Image", im4)
	# cv2.waitKey(0)

	return im4



def getPerspKeypoints(rgbFile1, rgbFile2, HFile1, HFile2, model, device):
	if HFile1 is None:
		igp1, img1 = read_and_process_image(rgbFile1, H=None)
	else:
		H1 = np.load(HFile1)
		igp1, img1 = read_and_process_image(rgbFile1, H=H1)

	c,h,w = igp1.shape

	if HFile2 is None:
		igp2, img2 = read_and_process_image(rgbFile2, H=None)
	else:
		H2 = np.load(HFile2)
		igp2, img2 = read_and_process_image(rgbFile2, H=H2)

	feat1 = extractSingle(igp1, model, device)
	feat2 = extractSingle(igp2, model, device)

	matches = mnn_matcher(
			torch.from_numpy(feat1['descriptors']).to(device=device),
			torch.from_numpy(feat2['descriptors']).to(device=device),
		)
	pos_a = feat1["keypoints"][matches[:, 0], : 2]
	pos_b = feat2["keypoints"][matches[:, 1], : 2]

	H, inliers = pydegensac.findHomography(pos_a, pos_b, 8.0, 0.99, 10000)
	pos_a = pos_a[inliers]
	pos_b = pos_b[inliers]

	inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in pos_a]
	inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in pos_b]
	placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(pos_a))]

	image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None, matchColor=[0, 255, 0])
	image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

	#### Visualization ####
	# cv2.imshow('Matches', image3)
	# cv2.waitKey()

	if HFile1 is None:
		return pos_a, pos_b, image3, image3

	orgSrc, orgDst = orgKeypoints(pos_a, pos_b, H1, H2)
	matchImg = drawOrg(cv2.imread(rgbFile1), cv2.imread(rgbFile2), orgSrc, orgDst) # Reproject matches to perspective View

	return orgSrc, orgDst, matchImg, image3

	

###### Ensemble
def read_and_process_image(img_path, resize=None, H=None, h=None, w=None, preprocessing='caffe'):
	img1 = Image.open(img_path)
	if resize:
		img1 = img1.resize(resize)
	if(img1.mode != 'RGB'):
		img1 = img1.convert('RGB')
	img1 = np.array(img1)
	if H is not None:
		img1 = cv2.warpPerspective(img1, H, dsize=(400, 400))
		# cv2.imshow("Image", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
		# cv2.waitKey(0)
	igp1 = torch.from_numpy(preprocess_image(img1, preprocessing=preprocessing).astype(np.float32))
	return igp1, img1

def mnn_matcher_scorer(descriptors_a, descriptors_b, k=np.inf):
	device = descriptors_a.device
	sim = descriptors_a @ descriptors_b.t()
	val1, nn12 = torch.max(sim, dim=1)
	val2, nn21 = torch.max(sim, dim=0)
	ids1 = torch.arange(0, sim.shape[0], device=device)
	mask = (ids1 == nn21[nn12])
	matches = torch.stack([ids1[mask], nn12[mask]]).t()
	remaining_matches_dist = val1[mask]
	return matches, remaining_matches_dist

def mnn_matcher(descriptors_a, descriptors_b):
	device = descriptors_a.device
	sim = descriptors_a @ descriptors_b.t()
	nn12 = torch.max(sim, dim=1)[1]
	nn21 = torch.max(sim, dim=0)[1]
	ids1 = torch.arange(0, sim.shape[0], device=device)
	mask = (ids1 == nn21[nn12])
	matches = torch.stack([ids1[mask], nn12[mask]])
	return matches.t().data.cpu().numpy()


def getPerspKeypointsEnsemble(model1, model2, rgbFile1, rgbFile2, HFile1, HFile2, device):
	if HFile1 is None:
		igp1, img1 = read_and_process_image(rgbFile1, H=None)
	else:
		H1 = np.load(HFile1)
		igp1, img1 = read_and_process_image(rgbFile1, H=H1)

	c,h,w = igp1.shape

	if HFile2 is None:
		igp2, img2 = read_and_process_image(rgbFile2, H=None)
	else:
		H2 = np.load(HFile2)
		igp2, img2 = read_and_process_image(rgbFile2, H=H2)

	with torch.no_grad():
		keypoints_a1, scores_a1, descriptors_a1 = process_multiscale(
			igp1.to(device).unsqueeze(0),
			model1,
			scales=[1]
		)
		keypoints_a1 = keypoints_a1[:, [1, 0, 2]]

		keypoints_a2, scores_a2, descriptors_a2 = process_multiscale(
			igp1.to(device).unsqueeze(0),
			model2,
			scales=[1]
		)
		keypoints_a2 = keypoints_a2[:, [1, 0, 2]]

		keypoints_b1, scores_b1, descriptors_b1 = process_multiscale(
			igp2.to(device).unsqueeze(0),
			model1,
			scales=[1]
		)
		keypoints_b1 = keypoints_b1[:, [1, 0, 2]]

		keypoints_b2, scores_b2, descriptors_b2 = process_multiscale(
			igp2.to(device).unsqueeze(0),
			model2,
			scales=[1]
		)
		keypoints_b2 = keypoints_b2[:, [1, 0, 2]]

	# calculating matches for both models
	matches1, dist_1 = mnn_matcher_scorer(
		torch.from_numpy(descriptors_a1).to(device=device),
		torch.from_numpy(descriptors_b1).to(device=device),
#                 len(matches1)
	)
	matches2, dist_2 = mnn_matcher_scorer(
		torch.from_numpy(descriptors_a2).to(device=device),
		torch.from_numpy(descriptors_b2).to(device=device),
#                 len(matches1)
	)

	full_matches = torch.cat([matches1, matches2])
	full_dist = torch.cat([dist_1, dist_2])
	assert len(full_dist)==(len(dist_1)+len(dist_2)), "something wrong"

	k_final = len(full_dist)//2
	# k_final = len(full_dist)
	# k_final = max(len(dist_1), len(dist_2))
	top_k_mask = torch.topk(full_dist, k=k_final)[1]
	first = []
	second = []

	for valid_id in top_k_mask:
		if valid_id<len(dist_1):
			first.append(valid_id)
		else:
			second.append(valid_id-len(dist_1))
	# final_matches = full_matches[top_k_mask]

	matches1 = matches1[torch.tensor(first, device=device).long()].data.cpu().numpy()
	matches2 = matches2[torch.tensor(second, device=device).long()].data.cpu().numpy()

	pos_a1 = keypoints_a1[matches1[:, 0], : 2]
	pos_b1 = keypoints_b1[matches1[:, 1], : 2]

	pos_a2 = keypoints_a2[matches2[:, 0], : 2]
	pos_b2 = keypoints_b2[matches2[:, 1], : 2]

	pos_a = np.concatenate([pos_a1, pos_a2], 0)
	pos_b = np.concatenate([pos_b1, pos_b2], 0)

	# pos_a, pos_b, inliers = apply_ransac(pos_a, pos_b)
	H, inliers = pydegensac.findHomography(pos_a, pos_b, 8.0, 0.99, 10000)
	pos_a = pos_a[inliers]
	pos_b = pos_b[inliers]

	inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in pos_a]
	inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in pos_b]
	placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(pos_a))]

	image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None, matchColor=[0, 255, 0])
	image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
	# cv2.imshow('Matches', image3)
	# cv2.waitKey()


	orgSrc, orgDst = orgKeypoints(pos_a, pos_b, H1, H2)
	matchImg = drawOrg(cv2.imread(rgbFile1), cv2.imread(rgbFile2), orgSrc, orgDst)

	return orgSrc, orgDst, matchImg, image3


if __name__ == '__main__':
	WEIGHTS = '../models/rord.pth'
	
	srcR = argv[1]
	trgR = argv[2]
	srcH = argv[3]
	trgH = argv[4]

	orgSrc, orgDst = getPerspKeypoints(srcR, trgR, srcH, trgH, WEIGHTS, ('gpu'))
