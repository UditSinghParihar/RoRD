import numpy as np
import argparse
import copy
import open3d as o3d
from sys import argv
from PIL import Image
import math
from extractMatchTop import getPerspKeypoints, getPerspKeypoints2, siftMatching, super_point_matcher
import os
import pandas as pd


import torch
from lib.model_test import D2Net

#### Cuda ####
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

#### Argument Parsing ####
parser = argparse.ArgumentParser(description='RoRD ICP evaluation')

parser.add_argument(
    '--rgb_csv', type=str, required=True,
    help='path to the csv file containing rgb images of query-database pairs'
)
parser.add_argument(
    '--depth_csv', type=str, required=True,
    help='path to the csv file containing depth files of query-database pairs'
)

parser.add_argument(
    '--output_dir', type=str, default='/scratch/udit/realsense/dataVO/data5/RT_rord/',
    help='output directory for RT estimates'
)

parser.add_argument(
    '--model_rord', type=str,
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
    '--superpoint', action='store_true',
    help='SuperPoint evaluation'
)

parser.add_argument(
    '--camera_file', type=str, default='/home/udit/d2-net/camera.txt',
    help='path to the camera intrinsics file. In order: focal_x, focal_y, center_x, center_y, scaling_factor.'
)

parser.add_argument(
    '--viz', action='store_true',
    help='visualize the pointcloud registrations'
)

args = parser.parse_args()


def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	# source_temp.paint_uniform_color([1, 0.706, 0])
	# target_temp.paint_uniform_color([0, 0.651, 0.929])
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
    focalX, focalY, centerX, centerY, scalingFactor = readCamera(args.camera_file)

    df_rgb = pd.read_csv(args.rgb_csv)
    df_dep = pd.read_csv(args.depth_csv)

    model1 = D2Net(model_file=args.model_d2).to(device)
    model2 = D2Net(model_file=args.model_rord).to(device)

    i = 0
    for im_q, dep_q in zip(df_rgb['query'], df_dep['query']):
        filter_list = []
        for im_d, dep_d in zip(df_rgb.iteritems(), df_dep.iteritems()):
            if im_d[0] == 'query':
                continue
            rgb_name_src = os.path.basename(im_q)
            H_name_src = os.path.splitext(rgb_name_src)[0] + '.npy'
            srcH = os.path.join(os.path.dirname(im_q), H_name_src)
            rgb_name_trg = os.path.basename(im_d[1][1])
            H_name_trg = os.path.splitext(rgb_name_trg)[0] + '.npy'
            trgH = os.path.join(os.path.dirname(im_d[1][1]), H_name_trg)

            if args.model_rord:
                srcPts, trgPts = getPerspKeypoints(im_q, im_d[1][1], srcH, trgH, model2, device)
            elif args.model_d2:
                srcPts, trgPts = getPerspKeypoints(im_q, im_d[1][1], srcH, trgH, model1, device)
            elif args.model_ens:
                srcPts, trgPts = getPerspKeypoints2(model1, model2, im_q, im_d[1][1], srcH, trgH, device)
            elif args.sift:
                srcPts, trgPts = siftMatching(im_q, im_d[1][1], srcH, trgH, device)
            elif args.superpoint:
                from SuperGluePretrainedNetwork.models.matching import Matching
                config = {
            		'superpoint': {
            			'nms_radius': 4,
            			'keypoint_threshold': 0.005,
            			'max_keypoints': 1024
            		},
            		'superglue': {
            			'weights': 'outdoor',
            			'sinkhorn_iterations': 20,
            			'match_threshold': 0.2,
            		}
            	}
                matching = Matching(config).eval().to(device)
                srcPts, trgPts = super_point_matcher(matching, im_q, im_d[1][1], srcH, trgH, device)

            if(isinstance(srcPts, list) == True):
                print(np.identity(4))
                filter_list.append(np.identity(4))
                continue


            srcPts = convertPts(srcPts)
            trgPts = convertPts(trgPts)

            srcCld, srcIdx, srcCor = getPointCloud(im_q, dep_q, srcPts)
            trgCld, trgIdx, trgCor = getPointCloud(im_d[1][1], dep_d[1][1], trgPts)

            srcSph = getSphere(srcCor)
            trgSph = getSphere(trgCor)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            srcSph.append(srcCld); srcSph.append(axis)
            trgSph.append(trgCld); trgSph.append(axis)

            corr = get3dCor(srcIdx, trgIdx)

            p2p = o3d.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(srcCld, trgCld, o3d.utility.Vector2iVector(corr))
            print(trans_init)
            filter_list.append(trans_init)

            if args.viz:
                o3d.visualization.draw_geometries(srcSph)
                o3d.visualization.draw_geometries(trgSph)
                draw_registration_result(srcCld, trgCld, trans_init)


        RT = np.stack(filter_list).transpose(1,2,0)
        dir_name = args.output_dir
        os.makedirs(dir_name, exist_ok=True)
        np.save(dir_name + str(i) + '.npy', RT)
        i+=1
        print('-----check-------', RT.shape)
