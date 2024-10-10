import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ[
  "TOKENIZERS_PARALLELISM"] = "true"  # If the process hangs, set this to false: https://github.com/huggingface/transformers/issues/5486.
os.environ['TMPDIR'] = '/lfs/local/0/fifty/tmp'
tempfile.tempdir = '/lfs/local/0/fifty/tmp'

import sys
import torch

import numpy as np

from PIL import Image

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.data.dataloader import get_dataloader
from src.train_utils.utils import train_parser, get_model

"""
The arguments don't matter here -- they're overwritten by the arguments in the saved model checkpoint, but some need
to be specified because they lack a default field. 

python src/reconstruction_fid.py \
      --epoch 100 \
      --lr 0.01 \
      --weight_decay 0 \
      --opt adam \
      --batch_size 100 \
      --n_accumulate 1 \
      --dropout 0.0 \
      --gpu 5 \
      --seed 0 \
      --save_dir test \
      --dataset imagenet \
      --model vqvae \
"""


def data_to_image(x, rec):
  def unnormalize(x):
    return (x * np.array([0.2686, 0.2613, 0.2758])) + np.array([0.4815, 0.4578, 0.4082])

  orig = x.transpose(1, 3).transpose(1, 2).numpy()
  orig = unnormalize(orig)  # Unnormalize the image.

  rec = rec.transpose(1, 3).transpose(1, 2)
  rec = rec.to(torch.float32).cpu().numpy()
  rec = unnormalize(rec)
  return orig, rec


@torch.no_grad()
def write_reconstructions(args, loader, models):
  device = torch.device(f'cuda:{args.gpu[0]}')
  for i, x in enumerate(loader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      # Get only the raw images if training on a dataset like ImageNet that returns (image, label) tuples.
      if len(x) == 2: x = x[0]
      init_x = x
      x = x.to(device)

      for j, (n, _, m) in enumerate(models):
        _, _, rec = m(x, return_rec=True)
        orig, rec = data_to_image(init_x, rec)
        for k in range(rec.shape[0]):
          image = Image.fromarray((rec[k] * 255).astype(np.uint8))
          image.save(f'fid/{n}/{i}_{k}.png')

          # # If on the first entry of the batch, save the original image.
          if j == 0:
            image = Image.fromarray((orig[k] * 255).astype(np.uint8))
            image.save(f'fid/orig/{i}_{k}.png')


def load_models():
  # (model name, path to model checkpoint).
  models = [
    ('vqvae', 'vqvae/100_4_euclidean_5e-05_16_0.0/20_model.pth/20_model.pth'),  # Euclidean distance VQ-VAE model.
    ('rot_vqvae', '100_4_rot_vqvae_euclidean_5e-05_16/20_model.pth'),  # Euclidean distance VQ-VAE model w/ rotation trick.
  ]
  rtn_models = []
  device = torch.device(f'cuda:{args.gpu[0]}')
  for (n, p) in models:
    vq_checkpoint = torch.load(p)
    vq_checkpoint['args'].gpu = args.gpu
    vq_model = get_model(vq_checkpoint['args'])
    vq_model = torch.nn.DataParallel(vq_model, device_ids=args.gpu)
    vq_model.to(device)
    vq_model.load_state_dict(vq_checkpoint['model'])
    vq_model.eval()
    rtn_models.append((n, p, vq_model))
  return rtn_models


if __name__ == '__main__':
  args = train_parser()

  # Get the different models.
  models = load_models()

  # Get valid dataloader.
  valid_dataloader = get_dataloader(args, split='val', distributed=False)
  write_reconstructions(args, valid_dataloader, models)
