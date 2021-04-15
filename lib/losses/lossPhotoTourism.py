import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import cv2
from sys import exit

import torch
import torch.nn.functional as F

from lib.utils import (
	grid_positions,
	upscale_positions,
	downscale_positions,
	savefig,
	imshow_image
)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')


def loss_function(
		model, batch, device, margin=1, safe_radius=4, scaling_steps=3, plot=False, plot_path=None
):
	output = model({
		'image1': batch['image1'].to(device),
		'image2': batch['image2'].to(device)
	})
	
	
	loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
	has_grad = False

	n_valid_samples = 0
	for idx_in_batch in range(batch['image1'].size(0)):
		# Network output
		dense_features1 = output['dense_features1'][idx_in_batch]
		c, h1, w1 = dense_features1.size()
		scores1 = output['scores1'][idx_in_batch].view(-1)

		dense_features2 = output['dense_features2'][idx_in_batch]
		_, h2, w2 = dense_features2.size()
		scores2 = output['scores2'][idx_in_batch]

		all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
		descriptors1 = all_descriptors1

		all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)

		fmap_pos1 = grid_positions(h1, w1, device)

		pos1 = batch['pos1'][idx_in_batch].to(device)
		pos2 = batch['pos2'][idx_in_batch].to(device)

		ids = idsAlign(pos1, device, h1, w1)

		fmap_pos1 = fmap_pos1[:, ids]
		descriptors1 = descriptors1[:, ids]
		scores1 = scores1[ids]

		# Skip the pair if not enough GT correspondences are available
		if ids.size(0) < 128:
			continue

		# Descriptors at the corresponding positions
		fmap_pos2 = torch.round(
			downscale_positions(pos2, scaling_steps=scaling_steps)
		).long()

		descriptors2 = F.normalize(
			dense_features2[:, fmap_pos2[0, :], fmap_pos2[1, :]],
			dim=0
		)
		positive_distance = 2 - 2 * (
			descriptors1.t().unsqueeze(1) @ descriptors2.t().unsqueeze(2)
		).squeeze()

		all_fmap_pos2 = grid_positions(h2, w2, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos2.unsqueeze(2).float() -
				all_fmap_pos2.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius

		distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)

		negative_distance2 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

		all_fmap_pos1 = grid_positions(h1, w1, device)
		position_distance = torch.max(
			torch.abs(
				fmap_pos1.unsqueeze(2).float() -
				all_fmap_pos1.unsqueeze(1)
			),
			dim=0
		)[0]
		is_out_of_safe_radius = position_distance > safe_radius

		distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)

		negative_distance1 = torch.min(
			distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
			dim=1
		)[0]

		diff = positive_distance - torch.min(
			negative_distance1, negative_distance2
		)

		scores2 = scores2[fmap_pos2[0, :], fmap_pos2[1, :]]

		loss = loss + (
			torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
			(torch.sum(scores1 * scores2) )
		)

		has_grad = True
		n_valid_samples += 1

		if plot and batch['batch_idx'] % batch['log_interval'] == 0:
			drawTraining(batch['image1'], batch['image2'], pos1, pos2, batch, idx_in_batch, output, save=True, plot_path=plot_path)

	if not has_grad:
		raise NoGradientError

	loss = loss / (n_valid_samples )

	return loss


def idsAlign(pos1, device, h1, w1):
	pos1D = downscale_positions(pos1, scaling_steps=3)
	row = pos1D[0, :]
	col = pos1D[1, :]

	ids = []

	for i in range(row.shape[0]):

		index = ((w1) * (row[i])) + (col[i])
		ids.append(index)

	ids = torch.round(torch.Tensor(ids)).long().to(device)

	return ids


def drawTraining(image1, image2, pos1, pos2, batch, idx_in_batch, output, save=False, plot_path="train_viz"):
	pos1_aux = pos1.cpu().numpy()
	pos2_aux = pos2.cpu().numpy()

	k = pos1_aux.shape[1]
	col = np.random.rand(k, 3)
	n_sp = 4
	plt.figure()
	plt.subplot(1, n_sp, 1)
	im1 = imshow_image(
		image1[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im1)
	plt.scatter(
		pos1_aux[1, :], pos1_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 2)
	plt.imshow(
		output['scores1'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 3)
	im2 = imshow_image(
		image2[0].cpu().numpy(),
		preprocessing=batch['preprocessing']
	)
	plt.imshow(im2)
	plt.scatter(
		pos2_aux[1, :], pos2_aux[0, :],
		s=0.25**2, c=col, marker=',', alpha=0.5
	)
	plt.axis('off')
	plt.subplot(1, n_sp, 4)
	plt.imshow(
		output['scores2'][idx_in_batch].data.cpu().numpy(),
		cmap='Reds'
	)
	plt.axis('off')

	if(save == True):
		savefig(plot_path+'/%s.%02d.%02d.%d.png' % (
			'train' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), dpi=300)
	else:
		plt.show()

	plt.close()

	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

	for i in range(0, pos1_aux.shape[1], 5):
		im1 = cv2.circle(im1, (pos1_aux[1, i], pos1_aux[0, i]), 1, (0, 0, 255), 2)
	for i in range(0, pos2_aux.shape[1], 5):
		im2 = cv2.circle(im2, (pos2_aux[1, i], pos2_aux[0, i]), 1, (0, 0, 255), 2)

	im3 = cv2.hconcat([im1, im2])

	for i in range(0, pos1_aux.shape[1], 5):
		im3 = cv2.line(im3, (int(pos1_aux[1, i]), int(pos1_aux[0, i])), (int(pos2_aux[1, i]) +  im1.shape[1], int(pos2_aux[0, i])), (0, 255, 0), 1)

	if(save == True):
		cv2.imwrite(plot_path+'/%s.%02d.%02d.%d.png' % (
			'train_corr' if batch['train'] else 'valid',
			batch['epoch_idx'],
			batch['batch_idx'] // batch['log_interval'],
			idx_in_batch
		), im3)
	else:
		cv2.imshow('Image', im3)
		cv2.waitKey(0)