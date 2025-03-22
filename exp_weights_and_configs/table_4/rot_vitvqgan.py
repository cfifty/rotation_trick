import torch

from einops import rearrange
from typing import Tuple

from enhancing.modules.stage1.vitvqgan import ViTVQ


class ROT_ViTVQ(ViTVQ):
  @staticmethod
  def get_rotation(u, q):
    b, d = u.shape
    w = (u + q) / torch.norm(u + q, dim=1, keepdim=True)
    return torch.eye(d).to(u.device) - 2 * torch.bmm(w.unsqueeze(-1), w.unsqueeze(1)) + 2 * torch.bmm(q.unsqueeze(-1),
                                                                                                      u.unsqueeze(1))
  def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # x: (b, 3, 256, 256)
    e = self.encoder(x)
    # h: (b, 1024, 32) // (b, 16*16, 32)
    e = self.pre_quant(e)
    b, t, w = e.shape
    # q: (b, 1024, 32)
    q, emb_loss, _ = self.quantizer(e)
    init_q = q

    # Apply a rotation to h so that it matches q.
    e = rearrange(e, 'b t c -> (b t) c')
    q = rearrange(q, 'b t c -> (b t) c')
    unit_e = e / torch.norm(e, dim=1, keepdim=True)
    unit_q = q / torch.norm(q, dim=1, keepdim=True)
    R = self.get_rotation(unit_e, unit_q).detach()
    q = torch.bmm(R, unit_e.unsqueeze(-1)).squeeze(-1) * torch.norm(q.detach(), dim=1, keepdim=True)
    q = rearrange(q, '(b t) c -> b t c', t=t)

    return q, emb_loss