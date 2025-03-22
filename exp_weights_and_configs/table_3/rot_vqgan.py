"""Adapted from https://github.com/CompVis/latent-diffusion"""

import torch
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from contextlib import contextmanager

from einops import rearrange

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config

from ldm.models.enhancing_autoencoders import SViTVQ, CustomVQInterface, ViTVQ, FSQ_ViTVQ
from ldm.models.taming_fsq import FSQ


class VQModel(pl.LightningModule):
  def __init__(self,
               ddconfig,
               lossconfig,
               n_embed,
               embed_dim,
               ckpt_path=None,
               ignore_keys=[],
               image_key="image",
               colorize_nlabels=None,
               monitor=None,
               batch_resize_range=None,
               scheduler_config=None,
               lr_g_factor=1.0,
               remap=None,
               sane_index_shape=False,  # tell vector quantizer to return indices as bhw
               use_ema=False,
               rot_trick=False,
               ref_trick=False,
               no_grad_scale=False,
               ):
    super().__init__()
    self.rot_trick = rot_trick
    self.ref_trick = ref_trick
    self.no_grad_scale = no_grad_scale
    self.embed_dim = embed_dim
    self.n_embed = n_embed
    self.image_key = image_key
    self.encoder = Encoder(**ddconfig)
    self.decoder = Decoder(**ddconfig)
    self.loss = instantiate_from_config(lossconfig)
    self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                    remap=remap,
                                    sane_index_shape=sane_index_shape)
    self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
    self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
    if colorize_nlabels is not None:
      assert type(colorize_nlabels) == int
      self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
    if monitor is not None:
      self.monitor = monitor
    self.batch_resize_range = batch_resize_range
    if self.batch_resize_range is not None:
      print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

    self.use_ema = use_ema
    if self.use_ema:
      self.model_ema = LitEma(self)
      print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    if ckpt_path is not None:
      self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    self.scheduler_config = scheduler_config
    self.lr_g_factor = lr_g_factor

  @contextmanager
  def ema_scope(self, context=None):
    if self.use_ema:
      self.model_ema.store(self.parameters())
      self.model_ema.copy_to(self)
      if context is not None:
        print(f"{context}: Switched to EMA weights")
    try:
      yield None
    finally:
      if self.use_ema:
        self.model_ema.restore(self.parameters())
        if context is not None:
          print(f"{context}: Restored training weights")

  def init_from_ckpt(self, path, ignore_keys=list()):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    keys = list(sd.keys())
    for k in keys:
      for ik in ignore_keys:
        if k.startswith(ik):
          print("Deleting key {} from state_dict.".format(k))
          del sd[k]
    missing, unexpected = self.load_state_dict(sd, strict=False)
    print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
      print(f"Missing Keys: {missing}")
      print(f"Unexpected Keys: {unexpected}")

  def on_train_batch_end(self, *args, **kwargs):
    if self.use_ema:
      self.model_ema(self)

  @staticmethod
  def get_rotation(u, q):
    b, d = u.shape
    w = (u + q) / torch.norm(u + q, dim=1, keepdim=True)
    return torch.eye(d).to(u.device) - 2 * torch.bmm(w.unsqueeze(-1), w.unsqueeze(1)) + 2 * torch.bmm(
      q.unsqueeze(-1), u.unsqueeze(1))

  @staticmethod
  def get_reflection(u, q):
    b, d = u.shape
    w = (u - q) / torch.norm(u - q, dim=1, keepdim=True)
    return torch.eye(d).to(u.device) - 2 * torch.bmm(w.unsqueeze(-1), w.unsqueeze(1))

  @torch.no_grad()
  def get_codes(self, x):
    x = self.encoder(x)
    x = self.quant_conv(x)
    quant, emb_loss, (p, e, idx) = self.quantize(x)
    return idx

  def encode(self, x):
    x = self.encoder(x)
    x = self.quant_conv(x)
    quant, emb_loss, info = self.quantize(x)
    if self.rot_trick:
      b, d, h, w = x.shape
      x = rearrange(x, 'b d h w -> (b h w) d')
      quantized = rearrange(quant, 'b d h w -> (b h w) d')
      unit_x = x / torch.norm(x, dim=1, keepdim=True)
      unit_q = (quantized / torch.norm(quantized, dim=1, keepdim=True))
      R = self.get_rotation(unit_x, unit_q).detach()
      q = torch.bmm(R, unit_x.unsqueeze(-1)).squeeze(-1) * torch.norm(quantized, dim=1, keepdim=True).detach()
      quant = rearrange(q, '(b h w) d -> b d h w', h=h, w=w)
    elif self.ref_trick:
      b, d, h, w = x.shape
      x = rearrange(x, 'b d h w -> (b h w) d')
      quantized = rearrange(quant, 'b d h w -> (b h w) d')
      unit_x = x / torch.norm(x, dim=1, keepdim=True)
      unit_q = (quantized / torch.norm(quantized, dim=1, keepdim=True))
      R = self.get_reflection(unit_x, unit_q).detach()
      q = torch.bmm(R, unit_x.unsqueeze(-1)).squeeze(-1) * torch.norm(quantized, dim=1, keepdim=True).detach()
      quant = rearrange(q, '(b h w) d -> b d h w', h=h, w=w)
    elif self.no_grad_scale:
      b, d, h, w = x.shape
      x = rearrange(x, 'b d h w -> (b h w) d')
      quantized = rearrange(quant, 'b d h w -> (b h w) d')
      unit_x = x / torch.norm(x, dim=1, keepdim=True)
      unit_q = (quantized / torch.norm(quantized, dim=1, keepdim=True))
      R = self.get_rotation(unit_x, unit_q).detach()
      q = torch.bmm(R, unit_x.unsqueeze(-1)).squeeze(-1)
      q = q + (quantized - q).detach()

      quant = rearrange(q, '(b h w) d -> b d h w', h=h, w=w)
    return quant, emb_loss, info

  def encode_to_prequant(self, x):
    h = self.encoder(x)
    h = self.quant_conv(h)
    return h

  def decode(self, quant):
    quant = self.post_quant_conv(quant)
    dec = self.decoder(quant)
    return dec

  def decode_code(self, code_b):
    quant_b = self.quantize.embed_code(code_b)
    dec = self.decode(quant_b)
    return dec

  def forward(self, input, return_pred_indices=False):
    quant, diff, (_, _, ind) = self.encode(input)
    dec = self.decode(quant)
    if return_pred_indices:
      return dec, diff, ind
    return dec, diff

  def get_input(self, batch, k):
    x = batch[k]
    if len(x.shape) == 3:
      x = x[..., None]
    x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
    if self.batch_resize_range is not None:
      lower_size = self.batch_resize_range[0]
      upper_size = self.batch_resize_range[1]
      if self.global_step <= 4:
        # do the first few batches with max size to avoid later oom
        new_resize = upper_size
      else:
        new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
      if new_resize != x.shape[2]:
        x = F.interpolate(x, size=new_resize, mode="bicubic")
      x = x.detach()
    return x

  def training_step(self, batch, batch_idx, optimizer_idx):
    # https://github.com/pytorch/pytorch/issues/37142
    # try not to fool the heuristics
    x = self.get_input(batch, self.image_key)
    xrec, qloss, ind = self(x, return_pred_indices=True)

    if optimizer_idx == 0:
      # autoencode
      aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                      last_layer=self.get_last_layer(), split="train")

      self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
      return aeloss

    if optimizer_idx == 1:
      # discriminator
      discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                          last_layer=self.get_last_layer(), split="train")
      self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
      return discloss

  def validation_step(self, batch, batch_idx):
    log_dict = self._validation_step(batch, batch_idx)
    with self.ema_scope():
      log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
    return log_dict

  def _validation_step(self, batch, batch_idx, suffix=""):
    x = self.get_input(batch, self.image_key)
    xrec, qloss, ind = self(x, return_pred_indices=True)
    aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                    self.global_step,
                                    last_layer=self.get_last_layer(),
                                    split="val" + suffix,
                                    )

    discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        )
    rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
    self.log(f"val{suffix}/rec_loss", rec_loss,
             prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    self.log(f"val{suffix}/aeloss", aeloss,
             prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
    del log_dict_ae[f"val{suffix}/rec_loss"]
    self.log_dict(log_dict_ae)
    self.log_dict(log_dict_disc)
    return self.log_dict

  def configure_optimizers(self):
    lr_d = self.learning_rate
    lr_g = self.lr_g_factor * self.learning_rate
    print("lr_d", lr_d)
    print("lr_g", lr_g)
    opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                              list(self.decoder.parameters()) +
                              list(self.quantize.parameters()) +
                              list(self.quant_conv.parameters()) +
                              list(self.post_quant_conv.parameters()),
                              lr=lr_g, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                lr=lr_d, betas=(0.5, 0.9))

    if self.scheduler_config is not None:
      scheduler = instantiate_from_config(self.scheduler_config)

      print("Setting up LambdaLR scheduler...")
      scheduler = [
        {
          'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
          'interval': 'step',
          'frequency': 1
        },
        {
          'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
          'interval': 'step',
          'frequency': 1
        },
      ]
      return [opt_ae, opt_disc], scheduler
    return [opt_ae, opt_disc], []

  def get_last_layer(self):
    return self.decoder.conv_out.weight

  def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
    log = dict()
    x = self.get_input(batch, self.image_key)
    x = x.to(self.device)
    if only_inputs:
      log["inputs"] = x
      return log
    xrec, _ = self(x)
    if x.shape[1] > 3:
      # colorize with random projection
      assert xrec.shape[1] > 3
      x = self.to_rgb(x)
      xrec = self.to_rgb(xrec)
    log["inputs"] = x
    log["reconstructions"] = xrec
    if plot_ema:
      with self.ema_scope():
        xrec_ema, _ = self(x)
        if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
        log["reconstructions_ema"] = xrec_ema
    return log

  def to_rgb(self, x):
    assert self.image_key == "segmentation"
    if not hasattr(self, "colorize"):
      self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    x = F.conv2d(x, weight=self.colorize)
    x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
    return x
