def get_model_params(dataset, f):
  # ImageNet reduced from (256, 256, 3) to (16, 16, embed_dim).
  if dataset == 'imagenet' and f == 8:
    channels = 3
    resolution = 256
    z_channels = 256
    embed_dim = 32
    n_embed = 1024
  elif dataset == 'imagenet' and f == 4:
    channels = 3
    resolution = 256
    z_channels = 3
    embed_dim = 3
    n_embed = 8192
  else:
    raise Exception(f'{dataset} setting for args.dataset is not supported.')
  return channels, resolution, z_channels, embed_dim, n_embed
