import numpy as np
import copy
import argparse
import os, sys
import open3d as o3d
from sys import argv
from PIL import Image
import math
import cv2
import torch

sys.path.append("../")
from lib.extractMatchTop import getPerspKeypoints, getPerspKeypointsEnsemble, siftMatching
from lib.model_test import D2Net

#### Cuda ####
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

#### Argument Parsing ####
parser = argparse.ArgumentParser(description='RoRD ICP evaluation')

parser.add_argument(
	'--rgb1', type=str, default = 'rgb/rgb2_1.jpg',
	help='path to the rgb image1'
)
parser.add_argument(
	'--rgb2', type=str, default = 'rgb/rgb2_2.jpg',
	help='path to the rgb image2'
)

parser.add_argument(
	'--depth1', type=str, default = 'depth/depth2_1.png',
	help='path to the depth image1'
)

parser.add_argument(
	'--depth2', type=str, default = 'depth/depth2_2.png',
	help='path to the depth image2'
)

parser.add_argument(
	'--model_rord', type=str, default = '../models/rord.pth',
	help='path to the RoRD model for evaluation'
)

parser.add_argument(
	'--model_d2', type=str,
	help='path to the vanilla D2-Net model for evaluation'
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
	'--camera_file', type=str, default='../configs/camera.txt',
	help='path to the camera intrinsics file. In order: focal_x, focal_y, center_x, center_y, scaling_factor.'
)

parser.add_argument(
	'--viz3d', action='store_true',
	help='visualize the pointcloud registrations'
)

args = parser.parse_args()

if args.model_ens: # Change default paths accordingly for ensemble
	model1_ens = '../../models/rord.pth'
	model2_ens = '../../models/d2net.pth'

def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.transform(transformation)

	target_temp += source_temp
	# print("Saved registered PointCloud.")
	# o3d.io.write_point_cloud("registered.pcd", target_temp)

	trgSph.append(source_temp); trgSph.append(target_temp)
	axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	axis2.transform(transformation)
	trgSph.append(axis1); trgSph.append(axis2)
	print("Showing registered PointCloud.")
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
				# print("Point found.")
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

	for ele in pts:
		if(ele is not None):
			sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
			sphere.paint_uniform_color([0.9, 0.2, 0])

			trans = np.identity(4)
			trans[0, 3] = ele[0]
			trans[1, 3] = ele[1]
			trans[2, 3] = ele[2]

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

	focalX, focalY, centerX, centerY, scalingFactor = readCamera(args.camera_file)

	rgb_name_src = os.path.basename(args.rgb1)
	H_name_src = os.path.splitext(rgb_name_src)[0] + '.npy'
	srcH = os.path.join(os.path.dirname(args.rgb1), H_name_src)
	rgb_name_trg = os.path.basename(args.rgb2)
	H_name_trg = os.path.splitext(rgb_name_trg)[0] + '.npy'
	trgH = os.path.join(os.path.dirname(args.rgb2), H_name_trg)

	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if use_cuda else 'cpu')
	model1 = D2Net(model_file=args.model_d2)
	model1 = model1.to(device)
	model2 = D2Net(model_file=args.model_rord)
	model2 = model2.to(device)

	if args.model_rord:
		srcPts, trgPts, matchImg, matchImgOrtho = getPerspKeypoints(args.rgb1, args.rgb2, srcH, trgH, model2, device)
	elif args.model_d2:
		srcPts, trgPts, matchImg, matchImgOrtho = getPerspKeypoints(args.rgb1, args.rgb2, srcH, trgH, model1, device)
	elif args.model_ens:
		model1 = D2Net(model_file=model1_ens)
		model1 = model1.to(device)
		model2 = D2Net(model_file=model2_ens)
		model2 = model2.to(device)
		srcPts, trgPts, matchImg, matchImgOrtho = getPerspKeypointsEnsemble(model1, model2, args.rgb1, args.rgb2, srcH, trgH, device)
	elif args.sift:
		srcPts, trgPts, matchImg, matchImgOrtho = siftMatching(args.rgb1, args.rgb2, srcH, trgH, device)

	#### Visualization ####
	print("\nShowing matches in perspective and orthographic view. Press q\n")
	cv2.imshow('Orthographic view', matchImgOrtho)
	cv2.imshow('Perspective view', matchImg)
	cv2.waitKey()

	srcPts = convertPts(srcPts)
	trgPts = convertPts(trgPts)

	srcCld, srcIdx, srcCor = getPointCloud(args.rgb1, args.depth1, srcPts)
	trgCld, trgIdx, trgCor = getPointCloud(args.rgb2, args.depth2, trgPts)

	srcSph = getSphere(srcCor)
	trgSph = getSphere(trgCor)
	axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
	srcSph.append(srcCld); srcSph.append(axis)
	trgSph.append(trgCld); trgSph.append(axis)

	corr = get3dCor(srcIdx, trgIdx)

	p2p = o3d.registration.TransformationEstimationPointToPoint()
	trans_init = p2p.compute_transformation(srcCld, trgCld, o3d.utility.Vector2iVector(corr))
	print("Transformation matrix: \n", trans_init)

	if args.viz3d:
		# o3d.visualization.draw_geometries(srcSph)
		# o3d.visualization.draw_geometries(trgSph)

		draw_registration_result(srcCld, trgCld, trans_init)
