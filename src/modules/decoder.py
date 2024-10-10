"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers"""
import torch
import sys

import numpy as np
import torch.nn as nn

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.modules.blocks import ResnetBlock, AttnBlock, nonlinearity, Normalize, Upsample


class Decoder(nn.Module):
  def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
               attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
               resolution, z_channels, give_pre_end=False, **ignorekwargs):
    super().__init__()
    self.ch = ch
    self.temb_ch = 0
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.resolution = resolution
    self.in_channels = in_channels
    self.give_pre_end = give_pre_end

    # compute in_ch_mult, block_in and curr_res at lowest res
    in_ch_mult = (1,) + tuple(ch_mult)
    block_in = ch * ch_mult[self.num_resolutions - 1]
    curr_res = resolution // 2 ** (self.num_resolutions - 1)
    self.z_shape = (1, z_channels, curr_res, curr_res)
    print("Working with z of shape {} = {} dimensions.".format(
      self.z_shape, np.prod(self.z_shape)))

    # z to block_in
    self.conv_in = torch.nn.Conv2d(z_channels,
                                   block_in,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)
    self.mid.attn_1 = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                   out_channels=block_in,
                                   temb_channels=self.temb_ch,
                                   dropout=dropout)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = ch * ch_mult[i_level]
      for i_block in range(self.num_res_blocks + 1):
        block.append(ResnetBlock(in_channels=block_in,
                                 out_channels=block_out,
                                 temb_channels=self.temb_ch,
                                 dropout=dropout))
        block_in = block_out
        if curr_res in attn_resolutions:
          attn.append(AttnBlock(block_in))
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in, resamp_with_conv)
        curr_res = curr_res * 2
      self.up.insert(0, up)  # prepend to get consistent order

    # end
    self.norm_out = Normalize(block_in)
    self.conv_out = torch.nn.Conv2d(block_in,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

  def forward(self, z):
    # assert z.shape[1:] == self.z_shape[1:]
    self.last_z_shape = z.shape

    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = self.up[i_level].block[i_block](h, temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    # end
    if self.give_pre_end:
      return h

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    return h
