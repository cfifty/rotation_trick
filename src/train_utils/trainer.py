import os
import sys
import torch

import numpy as np

import torch.distributed as dist

from tqdm import tqdm

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.train_utils.utils import get_logger, get_opt
from src.train_utils.wandb_utils import log_trainer_metrics


class Train_Manager:

  def __init__(self, args, train_fn, valid_fn, logging_fn):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    outer_dir = args.save_dir
    name = f'{args.epoch}_{args.f}_{args.model}_{args.codebook}_{args.lr}_{args.batch_size}'
    os.makedirs(f'{outer_dir}/{name}', exist_ok=True)

    self.logger = get_logger(f'{outer_dir}/{name}/train.log')
    self.save_path = f'{outer_dir}/{name}'

    self.logger.info('display all the hyper-parameters in args:')
    for arg in vars(args):
      value = getattr(args, arg)
      if value is not None:
        self.logger.info('%s: %s' % (str(arg), str(value)))
    self.logger.info('------------------------')
    self.args = args
    self.train_fn = train_fn
    self.valid_fn = valid_fn
    self.logging_fn = logging_fn
    self.scaler = torch.cuda.amp.GradScaler()

  def train(self, model):
    optimizer, scheduler = get_opt(model, self.args)
    best_epoch = 0
    best_val_loss = 1e6

    self.logger.info("start training!")
    for e in tqdm(range(self.args.epoch)):
      best_val_loss, best_epoch = self.epoch_loop(e, model, optimizer, scheduler, e, best_val_loss, best_epoch)
    self.log_end_of_training(best_epoch, best_val_loss)

  def resume(self, model, optimizer, scheduler, epoch):
    best_epoch = epoch+1
    best_val_loss = 1e6

    self.logger.info("start training!")
    for e in tqdm(range(epoch + 1, self.args.epoch)):
      best_val_loss, best_epoch = self.epoch_loop(e, model, optimizer, scheduler, e, best_val_loss, best_epoch)
    self.log_end_of_training(best_epoch, best_val_loss)

  def epoch_loop(self, epoch, model, optimizer, scheduler, e, best_val_loss, best_epoch):
    rec_loss = val_loss = commit_loss = 0

    # Compute training stats.
    if not self.args.skip_training:
      rec_loss, commit_loss = self.train_fn(optimizer=optimizer, scaler=self.scaler, scheduler=scheduler, logger=self.logger)
      self.logger.info(f'reconstruction loss: {rec_loss:.3f}')
      self.logger.info(f'commitment loss: {commit_loss:.3f}')

    # Compute valid stats.
    if not self.args.skip_validation:
      with torch.no_grad():
        val_loss = self.valid_fn(scaler=self.scaler, logger=self.logger)
        self.logger.info(f'val_losses: {val_loss:.3f}')

    # Log loss values and generated images.
    if dist.get_rank() == 0:
      log_trainer_metrics(rec_loss, commit_loss, val_loss, scheduler, e)
      if not self.args.skip_logging:
        if self.logging_fn:
          for l in self.logging_fn:
            l(epoch=e, save_path=self.save_path)

    torch.save(
      {'args': self.args, 'model': model.state_dict(), 'epoch': e, 'optimizer': optimizer.state_dict(),
       'scheduler': scheduler, 'best_val_loss': best_val_loss},
      f'{self.save_path}/{epoch}_model.pth')
    return best_val_loss, best_epoch

  def log_end_of_training(self, best_epoch, best_val_loss):
    self.logger.info('training finished!')
    self.logger.info('------------------------')
    self.logger.info(('the best epoch is %d/%d') % (best_epoch, self.args.epoch))
    self.logger.info(('the best val acc is %.3f') % (best_val_loss))

