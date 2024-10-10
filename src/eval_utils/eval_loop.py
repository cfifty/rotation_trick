import sys
import torch

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.train_utils.utils import get_device


def eval_loop(args, loader, model, world_size, scaler, logger, max_steps=4000):
  model.eval()
  device = get_device(model)

  avg_loss = 0
  for i, x in enumerate(loader):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
      # Get only the raw images if training on a dataset like ImageNet that returns (image, label) tuples.
      if len(x) == 2: x = x[0]
      x = x.to(device)
      rec_loss, _ = model(x)
      rec_loss = torch.mean(rec_loss)

      loss = rec_loss
      loss_value = loss.item()

    if i % 50 == 0:
      print(f'Eval loss at step {i}: {loss_value:.3f}')
    avg_loss += loss_value
    if i > max_steps:
      break
  avg_loss /= (i + 1)
  logger.info(f'average eval loss: {avg_loss:3f}')
  return avg_loss


