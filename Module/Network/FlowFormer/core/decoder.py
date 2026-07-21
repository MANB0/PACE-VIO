import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import coords_grid, bilinear_sampler
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine


from .gru import GMAUpdateBlock
from .gma import Attention

def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, H, W = img.size(0), img.size(2), img.size(3)     # img = (N, C, H, W)
    mean0   = coords_grid(N, H, W, img.device, img.dtype)
    mean1   = mean0.detach().clone()
    return mean0, mean1


class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim: int, v_dim: int, query_token_dim: int,
                 tgt_token_dim: int, num_heads: int =8,
                 proj_drop: float =0., dropout: float=0.):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5
        self.query_token_dim = query_token_dim

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q = nn.Linear(query_token_dim, qk_dim, bias=True)
        self.k = nn.Linear(tgt_token_dim, qk_dim, bias=True)
        self.v = nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim * 2, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )
        self.dim = qk_dim
    
    def forward(
        self,
        query: torch.Tensor,
        inp_key: torch.Tensor | None,
        inp_value: torch.Tensor | None,
        memory: torch.Tensor,
        query_coord: torch.Tensor
    ):
        """
        query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        k: torch.Tensor = self.k(memory) if inp_key is None else inp_key
        v: torch.Tensor = self.v(memory) if inp_value is None else inp_value

        coords_flat = query_coord.permute(0, 2, 3, 1).reshape(B * H1 * W1, 1, 2)
        query_coord_enc = LinearPositionEmbeddingSine(coords_flat, dim=self.dim)

        q         = self.norm1(query)
        q         = self.q(q + query_coord_enc)

        x         = self.multi_head_attn(q, k, v)
        x         = self.proj(torch.cat([x, query],dim=2))
        x         = query + self.proj_drop(x)

        x = x + self.ffn(self.norm2(x))
        return x, k, v


class MemoryDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg

        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_attend = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor | None, value: torch.Tensor | None,
                memory: torch.Tensor, coords1: torch.Tensor, size: tuple[int, int, int, int], query_latent_dim: int):
        """
            x:      [B*H1*W1, 1, C]
            memory: [B*H1*W1, H2'*W2', C]
            coords1 [B, 2, H2, W2]
            size: B, C, H1, W1
            1. Note that here coords0 and coords1 are in H2, W2 space.
               Should first convert it into H2', W2' space.
            2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(query, key, value, memory, coords1)
        B, C, H1, W1 = size
        # C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, query_latent_dim).permute(0, 3, 1, 2)
        return x_global, k, v


class MemoryDecoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        self.flow_token_encoder = nn.Sequential(
            nn.Conv2d(81 * cfg.cost_heads_num, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1)
        )
        self.proj = nn.Conv2d(256, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(cfg)
        
        self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
        self.att          = Attention(args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128)
        
        r = 4
        dx    = torch.linspace(-r, r, 2*r+1)
        dy    = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), dim=-1)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        self.register_buffer("delta", delta)
            
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow.shape
        
        mask = mask.view(N, 9, 8, 8, H, W).softmax(dim=1)
        up_flow = F.unfold(8 * flow, (3, 3), padding=1).view(N, C, 9, H, W)
        weighted = (mask.unsqueeze(1) * up_flow.unsqueeze(-3).unsqueeze(-3)).sum(dim=2)
        
        return weighted.permute(0, 1, 4, 2, 5, 3).reshape(N, C, 8 * H, 8 * W)

    def encode_flow_token(self, cost_maps, coords):
        """
            cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
            coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        centroid = coords.reshape(batch*h1*w1, 1, 1, 2)
        coords   = centroid + self.delta
        corr     = bilinear_sampler(cost_maps, coords)
        corr     = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def forward(
        self, cost_memory, context, cost_maps, query_latent_dim, flow_init=None
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        """
            memory: [B*H1*W1, H2'*W2', C]
            context: [B, D, H1, W1]
        """
        # cost_maps = data['cost_maps']
        coords0, coords1 = initialize_flow(context)

        if flow_init is not None:
            #print("[Using warm start]")
            coords1 = coords1 + flow_init

        flow_predictions = []

        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        attention = self.att(inp)

        size = (net.size(0), net.size(1), net.size(2), net.size(3)) # net.shape
        # key, value = None, None
        key: None | torch.Tensor = None
        value: None | torch.Tensor = None

        for _ in range(self.depth):
            coords1 = coords1.detach()

            cost_forward = self.encode_flow_token(cost_maps, coords1)

            query = self.flow_token_encoder(cost_forward)
            query = query.permute(0, 2, 3, 1).contiguous().view(size[0]*size[2]*size[3], 1, self.dim)
            cost_global, key, value = self.decoder_layer(query, key, value, cost_memory, coords1, size, query_latent_dim)
            corr = torch.cat([cost_global, cost_forward], dim=1)

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # flow = delta_flow
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)
        
        if self.training:
            return flow_predictions, None
        else:
            return [flow_predictions[-1]], coords1-coords0
