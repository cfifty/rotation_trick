import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # If the process hangs, set this to false: https://github.com/huggingface/transformers/issues/5486.
os.environ['TMPDIR'] = '/lfs/local/0/fifty/tmp'
tempfile.tempdir = '/lfs/local/0/fifty/tmp'

import sys
import torch
import wandb


import torch.distributed as dist
import torch.multiprocessing as mp

from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP


from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.data.dataloader import get_dataloader
from src.eval_utils.eval_loop import eval_loop
from src.train_utils.train_loop import train_loop
from src.train_utils.trainer import Train_Manager
from src.train_utils.utils import train_parser, get_model, log_codebook_usage, get_opt
from src.train_utils.wandb_utils import init_wandb



def resume_model_and_trainer(args, ckpt, rank, world_size):
  model = get_model(args)
  model = DDP(model.to(rank), device_ids=[rank])
  model.load_state_dict(ckpt['model'])

  # Set up the datasets.
  train_dataloader = get_dataloader(args, split='train')
  valid_dataloader = get_dataloader(args, split='val')

  # Set up the train/eval functions.
  train_fn = partial(train_loop, args=args, loader=train_dataloader, model=model,
                     world_size=world_size)
  valid_fn = partial(eval_loop, args=args, loader=valid_dataloader, model=model,
                     world_size=world_size)
  logging_fn = [
                # partial(reconstruct_latents, dataset=args.dataset, loader=valid_dataloader, model=model, num_plots=6),
                partial(log_codebook_usage, dataset=args.dataset, loader=valid_dataloader, model=model,
                        batch_size=args.batch_size),
                ]
  tm = Train_Manager(args, train_fn=train_fn, valid_fn=valid_fn, logging_fn=logging_fn)
  return model, tm

def get_model_and_trainer(args, rank, world_size):
  # Load the model. Note: using x GPUs will split the batch_size up x ways.
  model = get_model(args)
  model = DDP(model.to(rank), device_ids=[rank])

  # Set up the datasets.
  train_dataloader = get_dataloader(args, split='train')
  valid_dataloader = get_dataloader(args, split='val')

  # Set up the train/eval functions.
  train_fn = partial(train_loop, args=args, loader=train_dataloader, model=model,
                     world_size=world_size)
  valid_fn = partial(eval_loop, args=args, loader=valid_dataloader, model=model,
                     world_size=world_size)
  logging_fn = [
                # partial(reconstruct_latents, dataset=args.dataset, loader=valid_dataloader, model=model, num_plots=6),
                partial(log_codebook_usage, dataset=args.dataset, loader=valid_dataloader, model=model,
                        batch_size=args.batch_size),
                ]
  tm = Train_Manager(args, train_fn=train_fn, valid_fn=valid_fn, logging_fn=logging_fn)
  return model, tm

def setup(rank, world_size, port):
  # Initialize the process group
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = f'2950{port}'
  os.environ['WORLD_SIZE'] = str(world_size)
  os.environ['RANK'] = str(rank)
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

def train_model(rank, world_size):
  args = train_parser()
  setup(rank, world_size, args.port)
  torch.manual_seed(args.seed)
  if dist.get_rank() == 0:
    if args.wandb: init_wandb(args)
  if args.checkpoint != '':
    ckpt = torch.load(args.checkpoint)
    model, tm = resume_model_and_trainer(args, ckpt, rank, world_size)
    optimizer, scheduler = get_opt(model, args)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    tm.resume(model, optimizer, scheduler, ckpt['epoch'])
  else:
    model, tm = get_model_and_trainer(args, rank, world_size)
    tm.train(model)
  if args.wandb: wandb.finish()
  cleanup()


if __name__ == '__main__':
  world_size = torch.cuda.device_count()  # Number of GPUs -- change w/ CUDA_VISIBLE_DEVICES env. variable.
  mp.spawn(train_model,
           args=(world_size,),
           nprocs=world_size,
           join=True)
