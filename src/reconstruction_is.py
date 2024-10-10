from pytorch_image_generation_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

"""
python src/reconstruction_is.py
"""

fid_dirs = [
  'vqvae',
  'rot_vqvae',
]
for dir in fid_dirs:
  IS, IS_std = get_inception_score_from_directory(f'fid/{dir}')
  print(f'IS: {IS} from {dir}')


