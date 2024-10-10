import argparse
import logging
import sys
import torch
import wandb

import numpy as np

import matplotlib.pyplot as plt
import torch.optim as optim

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.vq_vae import VQVAE
from src.train_utils.schedulers import WarmupCosineFlatSchedule


def get_device(model):
  if isinstance(model.module, VQVAE):
    device = model.module.encoder.conv_in.weight.device
  else:
    device = model.module.vq_model.module.encoder.conv_in.weight.device
  return device


def get_model(args):
  if args.model in ['vqvae', 'dfp_vqvae', 'rot_vqvae', 'vhp_vqvae']:
    model = VQVAE(args)
  else:
    raise Exception(f'model {args.model} is not supported.')
  return model


def get_logger(filename):
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                datefmt='%m/%d %I:%M:%S')
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  fh = logging.FileHandler(filename, "a")
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  sh = logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger


def train_parser():
  parser = argparse.ArgumentParser()

  # Optimization arguments.
  parser.add_argument('--epoch', help='number of epochs to train', type=int)
  parser.add_argument('--lr', help='initial learning rate', type=float, default=6e-4)
  parser.add_argument('--warmup_iters', help='Number of steps for warmup of lr', type=int, default=2000)
  parser.add_argument('--decay_iters', help='Number of steps for cosine decay of lr', type=int, default=100000)
  parser.add_argument('--weight_decay', help='weight decay for optimizer', type=float, default=0.)
  parser.add_argument('--opt', help='optimizer', choices=['adam', 'sgd'])
  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--n_accumulate', help='Number of batches to accumulate before backward step.', type=int,
                      default=1)

  # Model arguments
  parser.add_argument('--model', help='The type of model to run', type=str, default='vqvae')
  parser.add_argument('--dropout', help='dropout for the model', type=float, default=0.0)
  parser.add_argument('--seed', help='random seed', type=int, default=42)
  parser.add_argument('--f', help='Image downsampling factor: 256 h -> 32 h is a downsample factor of 8.', type=int,
                      default=8)
  parser.add_argument('--codebook', help='Euclidean or Rotation codebook.', default='euclidean')
  parser.add_argument('--commit_weight', help='Commitment weight for the encoder to commit to a quantized vector',
                      type=float, default=1.0)
  parser.add_argument('--scale_factor', help='Scaling factor for dfp loss weight if that model is selected.',
                      type=float, default=1e6)

  # Misc. arguments.
  parser.add_argument('--dataset', help='Which dataset to use.', type=str, default='imagenet')
  parser.add_argument('--wandb', help='Log the training run with wandb or not.', action='store_true')
  parser.add_argument('--port', help='Port for DDP -- needs to be unique for each run.', type=int,
                      default=0)

  # Trainer arguments.
  parser.add_argument('--checkpoint', help='PATH for checkpoint file', type=str, default='')
  parser.add_argument('--gpu', help='gpu device', type=int, nargs='+', default=0)
  parser.add_argument('--save_dir', help='Where to save model files.', type=str, default='outputs')
  parser.add_argument('--skip_validation', help='Whether to validation or not.', action='store_true')
  parser.add_argument('--skip_logging', help='Whether to skip logging or not.', action='store_true')
  parser.add_argument('--skip_training',
                      help='Whether to skip training the model before going to eval --- useful for testing.',
                      action='store_true')
  args = parser.parse_args()
  return args


def get_opt(model, args):
  # If stage 1 training.
  if isinstance(model.module, VQVAE):
    if args.opt == 'adam':
      optimizer = optim.AdamW(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              betas=(0.9, 0.99))
    elif args.opt == 'sgd':
      optimizer = optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    else:
      raise Exception(f'optimizer {args.opt} not recognized.')
    min_lr_constant = 2  # Decays 1e-4 => 5e-5

  else:  # Otherwise, get the optimizer and learning rate schedule for stage 2.
    optimizer = optim.AdamW(model.module.seq_model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            betas=(0.9, 0.96))
    min_lr_constant = 45  # Decays 4.5e-4 => 1e-5

  scheduler = WarmupCosineFlatSchedule(optimizer, initial_lr=args.lr, warmup_iters=args.warmup_iters,
                                       decay_iters=args.decay_iters, min_lr_constant=min_lr_constant)
  return optimizer, scheduler


def reconstruct_latents(dataset, loader, model, epoch, save_path, num_plots=6):
  """
  Reconstruct an image from its latent embedding. Useful to ensure the model is properly training.
  """
  if dataset == 'imagenet':
    orig, rec, cmap = reconstruct_latent_from_imagenet(loader, model, num_plots=6)
  else:
    raise Exception(f'Dataset {dataset} is not recognized.')

  # Plot out the first num_plots predictions.
  fig, ax = plt.subplots(nrows=2, ncols=num_plots // 2, figsize=(10, 10))  # Create a figure with 4 subplots (2x2)
  for k in range(num_plots // 2):
    ax[0][k].imshow(orig[k], cmap=cmap, vmin=0, vmax=1, label=f'Orig_{k}')
    ax[1][k].imshow(rec[k], cmap=cmap, vmin=0, vmax=1, label=f'Rec_{k}')

  # Place a single legend on the figure
  handles, labels = ax[0][0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=num_plots // 2)
  plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the legend below the subplots

  plt.savefig(f'{save_path}/{epoch}_fig.png', dpi=300)
  plt.close()
  if wandb.run is not None:
    wandb.log({f'reconstructions': wandb.Image(f'{save_path}/{epoch}_fig.png')}, step=epoch)

  return


def log_codebook_usage(dataset, loader, model, epoch, save_path, batch_size):
  """
  Compute and log the codebook usage.

  The codebook usage is calculated as the percentage of used codes given a batch of 256 test images averaged
  over the entire test set.
  """
  model.eval()
  device = get_device(model)

  codes = []
  n_accumulate = 256 // batch_size  # Following VIT VQ-GAN.
  accumulate_count = 0
  usage_stats = []
  for i, x in enumerate(loader):
    with torch.no_grad():
      with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Get only the raw images if training on a dataset like ImageNet that returns (image, label) tuples.
        if len(x) == 2: x = x[0]
        x = x.to(device)
        codes.append(model.module.get_codes(x))

    accumulate_count += 1
    if accumulate_count == n_accumulate:
      codes = torch.concatenate(codes, dim=0)
      usage = torch.unique(codes).shape[0] / model.module.num_codes
      usage_stats.append(usage)
      accumulate_count = 0
      codes = []
  avg_usage = sum(usage_stats) / len(usage_stats)
  if wandb.run is not None:
    wandb.log({f'codebook_usage': avg_usage}, step=epoch)


def sample_images(dataset, loader, model, epoch, save_path, num_plots=6):
  """
  Sample an image from the learned distribution.
  """
  if dataset == 'imagenet':
    cmap = None
    samples = sample_image_from_imagenet(model, num_plots=num_plots)
  else:
    raise Exception(f'not yet supported.')

  # Plot out the first num_plots predictions.
  fig, ax = plt.subplots(nrows=2, ncols=num_plots // 2, figsize=(10, 10))  # Create a figure with 4 subplots (2x2)
  for k in range(num_plots // 2):
    ax[0][k].imshow(samples[k], cmap=cmap, vmin=0, vmax=1, label=f'Sample_{k}')
    ax[1][k].imshow(samples[num_plots // 2 + k], cmap=cmap, vmin=0, vmax=1, label=f'Sample_{num_plots // 2 + k}')

  # Place a single legend on the figure
  handles, labels = ax[0][0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=num_plots // 2)
  plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the legend below the subplots

  plt.savefig(f'{save_path}/{epoch}_fig.png', dpi=300)
  plt.close()
  if wandb.run is not None:
    wandb.log({f'samples': wandb.Image(f'{save_path}/{epoch}_fig.png')}, step=epoch)

  return


def reconstruct_latent_from_imagenet(loader, model, num_plots=6):
  def unnormalize(x):
    return (x * np.array([0.2686, 0.2613, 0.2758])) + np.array([0.4815, 0.4578, 0.4082])

  model.eval()
  device = get_device(model)
  for i, x in enumerate(loader):
    x = x[0]  # Get only the raw images if training on a dataset like ImageNet that returns (image, label) tuples.
    orig = x.transpose(1, 3).transpose(1, 2).numpy()
    orig = unnormalize(orig)  # Unnormalize the image.
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      with torch.no_grad():
        x = x.to(device)
        assert num_plots // 2 <= x.shape[0]
        _, _, rec = model(x, return_rec=True)
        # (b, c, h w) -> (b, h, w, c)
        rec = rec.transpose(1, 3).transpose(1, 2)
    rec = rec.to(torch.float32).cpu().numpy()
    rec = unnormalize(rec)
    return orig, rec, None


def sample_image_from_imagenet(model, num_plots=6):
  def unnormalize(x):
    return (x * np.array([0.2686, 0.2613, 0.2758])) + np.array([0.4815, 0.4578, 0.4082])

  model.eval()
  device = get_device(model)
  samples = model.module.decode(device, num_plots)
  samples = samples.transpose(1, 3).transpose(1, 2)
  samples = samples.to(torch.float32).cpu().numpy()
  samples = unnormalize(samples)
  return samples
