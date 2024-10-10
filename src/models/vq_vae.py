import sys
import torch

from einops import rearrange
from torch import nn

from torch.nn import functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from diffusers.models.autoencoders.vae import Encoder, Decoder
from taming.modules.diffusionmodules.model import Encoder, Decoder
from src.models.model_utils import get_model_params
from src.local_vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class VQVAE(nn.Module):
  def __init__(self, args):
    super().__init__()
    channels, resolution, z_channels, embed_dim, n_embed = get_model_params(args.dataset, args.f)
    self.args = args
    self.num_codes = n_embed
    self.cosine = (args.codebook == 'cosine')
    decay = 0.8  # Default value.

    if args.f == 8:
      self.encoder = Encoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                             out_ch=channels, ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16],
                             dropout=0.0)
      self.decoder = Decoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                             out_ch=channels,
                             ch=128, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)
    elif args.f == 4:
      self.encoder = Encoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                             out_ch=channels, ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[],
                             dropout=0.0)
      self.decoder = Decoder(double_z=False, z_channels=z_channels, resolution=resolution, in_channels=channels,
                             out_ch=channels,
                             ch=128, ch_mult=[1, 2, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)

    if args.codebook == 'cosine' or args.codebook == 'euclidean':
      # Note: ema_update=True and learnable_codebook=False, so will use ema updates to learn codebook vectors.
      self.vq = VectorQuantize(dim=embed_dim, codebook_size=n_embed, commitment_weight=args.commit_weight, decay=decay,
                               accept_image_fmap=True, use_cosine_sim=(args.codebook == 'cosine'),
                               threshold_ema_dead_code=0)
    else:
      raise Exception(f'codebook: {args.codebook} is not supported.')

    # Set up projections into and out of codebook.
    if args.codebook == 'cosine':
      self.pre_quant_proj = nn.Sequential(nn.Linear(z_channels, embed_dim),
                                          nn.LayerNorm(embed_dim)) if embed_dim != z_channels else nn.Identity()
    else:
      self.pre_quant_proj = nn.Sequential(
        nn.Linear(z_channels, embed_dim)) if embed_dim != z_channels else nn.Identity()
    self.post_quant_proj = nn.Linear(embed_dim, z_channels) if embed_dim != z_channels else nn.Identity()

  def get_codes(self, x):
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # VQ lookup.
    quantized, indices, _ = self.vq(x)
    return indices

  def decode(self, indices):
    q = self.vq.get_codes_from_indices(indices)
    if self.cosine:
      q = q / torch.norm(q, dim=1, keepdim=True)

    # Decode.
    x = self.post_quant_proj(q)
    x = rearrange(x, 'b (h w) c -> b c h w', h=16)
    x = self.decoder(x)
    return x

  def encode_forward(self, x):
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # VQ lookup.
    quantized, indices, _ = self.vq(x)
    return quantized

  def decoder_forward(self, q):
    if self.cosine:
      q = q / torch.norm(q, dim=1, keepdim=True)

    # Decode.
    x = rearrange(q, 'b c h w -> b h w c')
    x = self.post_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')
    x = self.decoder(x)
    return x

  @staticmethod
  def get_very_efficient_rotation(u, q, e):
    w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
    e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
      torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
    return e

  def forward(self, x, vhp=False, return_rec=False, double_fp=False, rot=False, loss_scale=1.0):
    init_x = x
    # Encode.
    x = self.encoder(x)
    x = rearrange(x, 'b c h w -> b h w c')
    x = self.pre_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')

    # ViT-VQGAN codebook: "We also apply l2 normalization on the encoded latent variables ze(x)
    # and codebook latent variables e."
    if self.cosine:
      x = x / torch.norm(x, dim=1, keepdim=True)

    e = x
    # VQ lookup.
    quantized, indices, commit_loss = self.vq(x)
    q = quantized

    # If using the rotation trick.
    if rot:
      b, c, h, w = x.shape
      x = rearrange(x, 'b c h w -> (b h w) c')
      quantized = rearrange(quantized, 'b c h w -> (b h w) c')
      pre_norm_q = self.get_very_efficient_rotation(x / (torch.norm(x, dim=1, keepdim=True) + 1e-6),
                                                    quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                    x.unsqueeze(1)).squeeze()
      quantized = pre_norm_q * (
              torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)).detach()
      quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)

    # If doing a double forward pass to get exact gradients, **do not** use the STE to update the encoder.
    if double_fp:
      quantized = quantized.detach()  # Remove STE estimator here.
    if self.cosine:
      quantized = quantized / torch.norm(quantized, dim=1, keepdim=True)

    # Use codebook ema: no embed loss.
    # emb_loss = F.mse_loss(quantized, x.detach())

    # Decode.
    x = rearrange(quantized, 'b c h w -> b h w c')
    x = self.post_quant_proj(x)
    x = rearrange(x, 'b h w c -> b c h w')
    x = self.decoder(x)
    rec = x
    rec_loss = F.mse_loss(init_x, x)

    # If using Hessian approximation of the gradients...
    if vhp:
      vhp_fn = lambda z: loss_scale * F.mse_loss(init_x, self.decoder_forward(z))
      df2 = torch.autograd.functional.vhp(func=vhp_fn, inputs=q, v=e - q)[1]
      e.register_hook(lambda grad: grad + df2.to(torch.bfloat16))

    # If doing a double forward-pass to compute exact gradients...
    if double_fp:
      x = e
      if self.cosine:
        x = x / torch.norm(x, dim=1, keepdim=True)

      # Decode.
      x = rearrange(x, 'b c h w -> b h w c')
      x.register_hook(lambda grad: grad * self.args.scale_factor)  # Scale gradient back up.
      x = self.post_quant_proj(x)
      x = rearrange(x, 'b h w c -> b c h w')
      x = self.decoder(x)

      fp2_rec_loss = 1 / self.args.scale_factor * F.mse_loss(init_x, x)
      rec_loss += fp2_rec_loss

    if return_rec:
      return rec_loss, commit_loss, rec
    return rec_loss, commit_loss
