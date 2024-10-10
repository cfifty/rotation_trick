"""Adapted from https://raw.githubusercontent.com/jeonsworld/ViT-pytorch/main/utils/scheduler.py"""
import logging
import math

from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


class WarmupCosineFlatSchedule(LambdaLR):
  """This is a bastardization of LambdaLR (should be a multiplicative factor multiplied by initial lr.

  But whatever."""
  def __init__(self, optimizer, initial_lr, warmup_iters, decay_iters, last_epoch=-1, min_lr_constant=2):
    self.initial_lr = initial_lr
    self.warmup_iters = warmup_iters
    self.decay_iters = decay_iters
    self.min_lr_constant = min_lr_constant  # Decay to the minimum learning rate = initial_lr / min_lr_constant
    super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

  def lr_lambda(self, step):
    # Linear warmup for warmup_iters.
    if step < self.warmup_iters:
      return self.initial_lr * (step / self.warmup_iters)
    # Return the minimum learning rate if step > lr_decay_iters.
    if step > self.decay_iters:
      return self.initial_lr / self.min_lr_constant
    # In between, use cosine decay down to minimum learning rate.
    decay_ratio = (step - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return self.initial_lr / self.min_lr_constant + coeff * (self.initial_lr - self.initial_lr / self.min_lr_constant)

  def get_lr(self):
    return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]