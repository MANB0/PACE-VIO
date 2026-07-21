import torch
import torch.nn as nn
from collections import OrderedDict

from .twins_svt import TwinsSVTLarge
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .utils import InputPadder


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)

        self.context_encoder = TwinsSVTLarge(pretrained=self.cfg.pretrain)

    def forward(self, image1, image2):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = (2 * image1) - 1.0
        image2 = (2 * image2) - 1.0

        context = self.context_encoder(image1)

        cost_memory, cost_maps = self.memory_encoder(image1, image2, context)

        flow_predictions = self.memory_decoder(cost_memory, context, cost_maps, self.cfg.query_latent_dim, flow_init=None)

        return flow_predictions

    @torch.no_grad()
    @torch.inference_mode()
    def inference(self, image1, image2):
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = self(image1, image2)

        flow_pre = padder.unpad(flow_pre[0])
        flow = flow_pre[0]
        return flow, torch.empty(0)

    def load_ddp_state_dict(self, ckpt: OrderedDict):
        cvt_ckpt = OrderedDict()
        for k in ckpt:
            if k.startswith("module."):
                cvt_ckpt[k[7:]] = ckpt[k]
            else:
                cvt_ckpt[k] = ckpt[k]
        self.load_state_dict(cvt_ckpt, strict=False)
