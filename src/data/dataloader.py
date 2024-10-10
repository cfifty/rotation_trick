import sys

from torch.utils.data import DataLoader, DistributedSampler

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.data.imagenet_dataset import imagenet_dataset


def get_dataloader(args, split, distributed=True):
  if args.dataset == 'imagenet':
    dataset = imagenet_dataset(split)
  else:
    raise Exception(f'{args.dataset} is not supported as a dataset class.')
  if distributed:  # Distributed uses DistributedSampler => it *does* shuffle.
    return DataLoader(dataset, batch_size=args.batch_size, sampler=DistributedSampler(dataset, shuffle=True),
                      shuffle=False, num_workers=10)
  return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)  # Validation => do not shuffle.
