
import argparse
import numpy as np
import os
import sys
sys.path.append("../")

import shutil

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import warnings

# from lib.dataset import MegaDepthDataset

from lib.exceptions import NoGradientError
from lib.loss import loss_function as orig_loss
from lib.losses.lossPhotoTourism import loss_function as ipr_loss
from lib.model import D2Net
from lib.dataloaders.datasetPhotoTourism_combined import PhotoTourismCombined


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

# Seed
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

# Argument parsing
parser = argparse.ArgumentParser(description='Training script')

parser.add_argument(
    '--dataset_path', type=str, default="/scratch/udit/phototourism/",
    help='path to the dataset'
)
# parser.add_argument(
#     '--scene_info_path', type=str, required=True,
#     help='path to the processed scenes'
# )

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)

parser.add_argument(
    '--model_file', type=str, default='models/d2_ots.pth',
    help='path to the full model'
)

parser.add_argument(
    '--num_epochs', type=int, default=10,
    help='number of training epochs'
)
parser.add_argument(
    '--lr', type=float, default=1e-3,
    help='initial learning rate'
)
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch size'
)
parser.add_argument(
    '--num_workers', type=int, default=16,
    help='number of workers for data loading'
)

parser.add_argument(
    '--use_validation', dest='use_validation', action='store_true',
    help='use the validation split'
)
parser.set_defaults(use_validation=False)

parser.add_argument(
    '--log_interval', type=int, default=250,
    help='loss logging interval'
)

parser.add_argument(
    '--log_file', type=str, default='log.txt',
    help='loss logging file'
)

parser.add_argument(
    '--plot', dest='plot', action='store_true',
    help='plot training pairs'
)
parser.set_defaults(plot=False)

parser.add_argument(
    '--checkpoint_directory', type=str, default='checkpoints',
    help='directory for training checkpoints'
)
parser.add_argument(
    '--checkpoint_prefix', type=str, default='d2',
    help='prefix for training checkpoints'
)

args = parser.parse_args()
print(args)

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_cuda=False
)
model = model.to(device)

# Optimizer
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
)

# Dataset
if args.use_validation:
    validation_dataset = PhotoTourismCombined(
        # scene_list_path='megadepth_utils/valid_scenes.txt',
        # scene_info_path=args.scene_info_path,
        base_path=args.dataset_path,
        train=False,
        preprocessing=args.preprocessing,
        pairs_per_scene=25
    )
    # validation_dataset.build_dataset()
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

training_dataset = PhotoTourismCombined(
    # scene_list_path='megadepth_utils/train_scenes.txt',
    # scene_info_path=args.scene_info_path,
    base_path=args.dataset_path,
    preprocessing=args.preprocessing
)
# training_dataset.build_dataset()

training_dataloader = DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)


# Define epoch function
def process_epoch(
        epoch_idx,
        model, loss_function, optimizer, dataloader, device,
        log_file, args, train=True, plot_path=None
):
    epoch_losses = []

    torch.set_grad_enabled(train)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (batch,method) in progress_bar:
        if train:
            optimizer.zero_grad()

        batch['train'] = train
        batch['epoch_idx'] = epoch_idx
        batch['batch_idx'] = batch_idx
        batch['batch_size'] = args.batch_size
        batch['preprocessing'] = args.preprocessing
        batch['log_interval'] = args.log_interval

        try:
            loss = loss_function[method](model, batch, device, plot=args.plot, plot_path=plot_path)
        except NoGradientError:
            # print("failed")
            continue

        current_loss = loss.data.cpu().numpy()[0]
        epoch_losses.append(current_loss)

        progress_bar.set_postfix(loss=('%.4f' % np.mean(epoch_losses)))

        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(dataloader), np.mean(epoch_losses)
            ))

        if train:
            loss.backward()
            optimizer.step()

    log_file.write('[%s] epoch %d - avg_loss: %f\n' % (
        'train' if train else 'valid',
        epoch_idx,
        np.mean(epoch_losses)
    ))
    log_file.flush()

    return np.mean(epoch_losses)


# Create the checkpoint directory
checkpoint_directory = os.path.join(args.checkpoint_directory, args.checkpoint_prefix)
if os.path.isdir(checkpoint_directory):
    print('[Warning] Checkpoint directory already exists.')
else:
    os.makedirs(checkpoint_directory, exist_ok=True)

# Open the log file for writing
log_file = os.path.join(checkpoint_directory,args.log_file)
if os.path.exists(log_file):
    print('[Warning] Log file already exists.')
log_file = open(log_file, 'a+')

# Create the folders for plotting if need be
plot_path=None
if args.plot:
    plot_path = os.path.join(checkpoint_directory,'train_vis')
    if os.path.isdir(plot_path):
        print('[Warning] Plotting directory already exists.')
    else:
        os.makedirs(plot_path, exist_ok=True)


# Initialize the history
train_loss_history = []
validation_loss_history = []
if args.use_validation:
    min_validation_loss = process_epoch(
        0,
        model, [orig_loss, ipr_loss], optimizer, validation_dataloader, device,
        log_file, args,
        train=False
    )

# Start the training
for epoch_idx in range(1, args.num_epochs + 1):
    # Process epoch
    train_loss_history.append(
        process_epoch(
            epoch_idx,
            model, [orig_loss, ipr_loss], optimizer, training_dataloader, device,
            log_file, args, train=True, plot_path=plot_path
        )
    )

    if args.use_validation:
        validation_loss_history.append(
            process_epoch(
                epoch_idx,
                model, [orig_loss, ipr_loss], optimizer, validation_dataloader, device,
                log_file, args,
                train=False
            )
        )

    # Save the current checkpoint
    checkpoint_path = os.path.join(
        checkpoint_directory,
        '%02d.pth' % (epoch_idx)
    )
    checkpoint = {
        'args': args,
        'epoch_idx': epoch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history
    }
    torch.save(checkpoint, checkpoint_path)
    if (
        args.use_validation and
        validation_loss_history[-1] < min_validation_loss
    ):
        min_validation_loss = validation_loss_history[-1]
        best_checkpoint_path = os.path.join(
            checkpoint_directory,
            '%s.best.pth' % args.checkpoint_prefix
        )
        shutil.copy(checkpoint_path, best_checkpoint_path)

# Close the log file
log_file.close()
