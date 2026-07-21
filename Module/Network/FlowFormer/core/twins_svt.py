import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .Twins.svt_large import twins_svt_large


class TwinsSVTLarge(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = twins_svt_large(pretrained=pretrained)

        del self.svt.norm
        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
    
    def forward(self, x):
        B, layer = x.shape[0], 2
        use_ckpt = torch.is_grad_enabled()
        
        for i, (embed, drop, blocks, pos_blk) in enumerate(zip(
            self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block
        )):
            x, size = embed(x)
            x = drop(x)
            
            for j, blk in enumerate(blocks):
                x = grad_checkpoint(blk, x, size, use_reentrant=False) if use_ckpt else blk(x, size)
                if j==0:
                    x = grad_checkpoint(pos_blk, x, size, use_reentrant=False) if use_ckpt else pos_blk(x, size)
            
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            
            if i == layer-1:
                break
        
        return x
