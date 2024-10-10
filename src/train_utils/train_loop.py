import copy
import sys
import torch

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.train_utils.utils import get_device


def train_loop(args, loader, model, world_size, optimizer, scaler, scheduler, logger, max_steps=10000):
  max_steps = int(
    max_steps / (args.batch_size * world_size / 16))  # Hack to ensure backwards consistency w.r.t. lr schedule :/
  model.train()
  device = get_device(model)
  optimizer.zero_grad()

  avg_rec_loss = accumulate_count = avg_commit_loss = 0
  for i, x in enumerate(loader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      # Get only the raw images if training on a dataset like ImageNet that returns (image, label) tuples.
      if len(x) == 2: x = x[0]

      x = x.to(device)
      rec_loss, vq_loss = model(x, vhp='vhp' in args.model, double_fp='dfp' in args.model, rot='rot' in args.model,
                                loss_scale=scaler.get_scale())

      rec_loss = torch.mean(rec_loss)
      vq_loss = torch.mean(vq_loss)

      loss = rec_loss + vq_loss
      rec_val = rec_loss.item()
      commit_val = vq_loss.item()
      scaler.scale(loss).backward()

      accumulate_count += 1

    if accumulate_count == args.n_accumulate:
      accumulate_count = 0
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      scaler.step(optimizer)  # Update parameters w.r.t. optimizer values.

      scaler.update()  # Update the scale factor for the next iteration.
      scheduler.step()  # Update the scheduler.
      optimizer.zero_grad()  # Zero out gradient attributes for all parameters.

    if i % (5 * args.n_accumulate) == 0:
      print(f'Train loss at step {i // args.n_accumulate}: {rec_val + commit_val:.3f}')
    avg_rec_loss += rec_val
    avg_commit_loss += commit_val
    if i > max_steps:
      break
  avg_rec_loss /= (i + 1)
  avg_commit_loss /= (i + 1)
  logger.info(f'average train loss: {avg_rec_loss + avg_commit_loss:3f}')
  return avg_rec_loss, avg_commit_loss
