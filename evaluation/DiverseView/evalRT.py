import numpy as np
import argparse
import copy
import os, sys
import open3d as o3d
from sys import argv, exit
from PIL import Image
import math
from tqdm import tqdm
import cv2


sys.path.append("../../")

from lib.extractMatchTop import getPerspKeypoints, getPerspKeypointsEnsemble, siftMatching
import pandas as pd


import torch
from lib.model_test import D2Net

#### Cuda ####
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

#### Argument Parsing ####
parser = argparse.ArgumentParser(description='RoRD ICP evaluation on a DiverseView dataset sequence.')

parser.add_argument('--dataset', type=str, default='/scratch/udit/realsense/RoRD_data/preprocessed/', 
	help='path to the dataset folder')

parser.add_argument('--sequence', type=str, default='data1')

parser.add_argument(
	'--output_dir', type=str, default='out',
	help='output directory for RT estimates'
)

parser.add_argument(
	'--model_rord', type=str, help='path to the RoRD model for evaluation'
)

parser.add_argument(
	'--model_d2', type=str, help='path to the vanilla D2-Net model for evaluation'
)

parser.add_argument(
	'--model_ens', action='store_true',
	help='ensemble model of RoRD + D2-Net'
)

parser.add_argument(
	'--sift', action='store_true',
	help='Sift'
)

parser.add_argument(
	'--viz3d', action='store_true',
	help='visualize the pointcloud registrations'
)

parser.add_argument(
	'--log_interval', type=int, default=9,
	help='Matched image logging interval'
)

parser.add_argument(
	'--camera_file', type=str, default='../../configs/camera.txt',
	help='path to the camera intrinsics file. In order: focal_x, focal_y, center_x, center_y, scaling_factor.'
)

parser.add_argument(
	'--persp', action='store_true', default=False,
	help='Feature matching on perspective images.'
)

parser.set_defaults(fp16=False)
args = parser.parse_args()


if args.model_ens: # Change default paths accordingly for ensemble
	model1_ens = '../../models/rord.pth'
	model2_ens = '../../models/d2net.pth'

def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.transform(transformation)
	trgSph.append(source_temp); trgSph.append(target_temp)
	axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2.transform(transformation)
	trgSph.append(axis1); trgSph.append(axis2)
	o3d.visualization.draw_geometries(trgSph)

def readDepth(depthFile):
	depth = Image.open(depthFile)
	if depth.mode != "I":
		raise Exception("Depth image is not in intensity format")

	return np.asarray(depth)

def readCamera(camera):
	with open (camera, "rt") as file:
		contents = file.read().split()

	focalX = float(contents[0])
	focalY = float(contents[1])
	centerX = float(contents[2])
	centerY = float(contents[3])
	scalingFactor = float(contents[4])

	return focalX, focalY, centerX, centerY, scalingFactor


def getPointCloud(rgbFile, depthFile, pts):
	thresh = 15.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	points = []
	colors = []

	corIdx = [-1]*len(pts)
	corPts = [None]*len(pts)
	ptIdx = 0

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY

			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

			if((u, v) in pts):
				index = pts.index((u, v))
				corIdx[index] = ptIdx
				corPts[index] = (X, Y, Z)

			ptIdx = ptIdx+1

	points = np.asarray(points)
	colors = np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)

	return pcd, corIdx, corPts


def convertPts(A):
	X = A[0]; Y = A[1]

	x = [];	y = []

	for i in range(len(X)):
		x.append(int(float(X[i])))

	for i in range(len(Y)):
		y.append(int(float(Y[i])))

	pts = []
	for i in range(len(x)):
		pts.append((x[i], y[i]))

	return pts


def getSphere(pts):
	sphs = []

	for element in pts:
		if(element is not None):
			sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
			sphere.paint_uniform_color([0.9, 0.2, 0])

			trans = np.identity(4)
			trans[0, 3] = element[0]
			trans[1, 3] = element[1]
			trans[2, 3] = element[2]

			sphere.transform(trans)
			sphs.append(sphere)

	return sphs


def get3dCor(src, trg):
	corr = []

	for sId, tId in zip(src, trg):
		if(sId != -1 and tId != -1):
			corr.append((sId, tId))

	corr = np.asarray(corr)

	return corr

if __name__ == "__main__":
	camera_file = args.camera_file
	rgb_csv = args.dataset + args.sequence + '/rtImagesRgb.csv'
	depth_csv = args.dataset + args.sequence + '/rtImagesDepth.csv'

	os.makedirs(os.path.join(args.output_dir, 'vis'), exist_ok=True)
	dir_name = args.output_dir
	os.makedirs(args.output_dir, exist_ok=True)

	focalX, focalY, centerX, centerY, scalingFactor = readCamera(camera_file)

	df_rgb = pd.read_csv(rgb_csv)
	df_dep = pd.read_csv(depth_csv)

	model1 = D2Net(model_file=args.model_d2).to(device)
	model2 = D2Net(model_file=args.model_rord).to(device)

	queryId = 0
	for im_q, dep_q in tqdm(zip(df_rgb['query'], df_dep['query']), total=df_rgb.shape[0]):
		filter_list = []
		dbId = 0
		for im_d, dep_d in tqdm(zip(df_rgb.iteritems(), df_dep.iteritems()), total=df_rgb.shape[1]):
			if im_d[0] == 'query':
				continue
			rgb_name_src = os.path.basename(im_q)
			H_name_src = os.path.splitext(rgb_name_src)[0] + '.npy'
			srcH = args.dataset + args.sequence + '/rgb/' + H_name_src
			rgb_name_trg = os.path.basename(im_d[1][1])
			H_name_trg = os.path.splitext(rgb_name_trg)[0] + '.npy'
			trgH = args.dataset + args.sequence + '/rgb/' + H_name_trg

			srcImg = srcH.replace('.npy', '.jpg')
			trgImg = trgH.replace('.npy', '.jpg')

			if args.model_rord:
				if args.persp:
					srcPts, trgPts, matchImg, _ = getPerspKeypoints(srcImg, trgImg, HFile1=None, HFile2=None, model=model2, device=device)
				else:
					srcPts, trgPts, matchImg, _ = getPerspKeypoints(srcImg, trgImg, srcH, trgH, model2, device)
			
			elif args.model_d2:
				if args.persp:
					srcPts, trgPts, matchImg, _ = getPerspKeypoints(srcImg, trgImg, HFile1=None, HFile2=None, model=model2, device=device)
				else:
					srcPts, trgPts, matchImg, _ = getPerspKeypoints(srcImg, trgImg, srcH, trgH, model1, device)
			
			elif args.model_ens:
				model1 = D2Net(model_file=model1_ens)
				model1 = model1.to(device)
				model2 = D2Net(model_file=model2_ens)
				model2 = model2.to(device)
				srcPts, trgPts, matchImg = getPerspKeypointsEnsemble(model1, model2, srcImg, trgImg, srcH, trgH, device)
			
			elif args.sift:
				if args.persp:
					srcPts, trgPts, matchImg, _ = siftMatching(srcImg, trgImg, HFile1=None, HFile2=None, device=device)
				else:
					srcPts, trgPts, matchImg, _ = siftMatching(srcImg, trgImg, srcH, trgH, device)

			if(isinstance(srcPts, list) == True):
				print(np.identity(4))
				filter_list.append(np.identity(4))
				continue


			srcPts = convertPts(srcPts)
			trgPts = convertPts(trgPts)

			depth_name_src = os.path.dirname(os.path.dirname(args.dataset)) + '/' + dep_q
			depth_name_trg = os.path.dirname(os.path.dirname(args.dataset)) + '/' + dep_d[1][1]

			srcCld, srcIdx, srcCor = getPointCloud(srcImg, depth_name_src, srcPts)
			trgCld, trgIdx, trgCor = getPointCloud(trgImg, depth_name_trg, trgPts)

			srcSph = getSphere(srcCor)
			trgSph = getSphere(trgCor)
			axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
			srcSph.append(srcCld); srcSph.append(axis)
			trgSph.append(trgCld); trgSph.append(axis)

			corr = get3dCor(srcIdx, trgIdx)

			p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
			trans_init = p2p.compute_transformation(srcCld, trgCld, o3d.utility.Vector2iVector(corr))
			# print(trans_init)
			filter_list.append(trans_init)

			if args.viz3d:
				o3d.visualization.draw_geometries(srcSph)
				o3d.visualization.draw_geometries(trgSph)
				draw_registration_result(srcCld, trgCld, trans_init)

			if(dbId%args.log_interval == 0):
				cv2.imwrite(os.path.join(args.output_dir, 'vis') + "/matchImg.%02d.%02d.jpg"%(queryId, dbId//args.log_interval), matchImg)
			dbId += 1


		RT = np.stack(filter_list).transpose(1,2,0)

		np.save(os.path.join(dir_name, str(queryId) + '.npy'), RT)
		queryId += 1
		print('-----check-------', RT.shape)
