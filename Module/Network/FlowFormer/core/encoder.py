import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .twins import Block
from .utils import coords_grid
from .twins_svt import TwinsSVTLarge
from .attention import BroadMultiHeadAttention, MultiHeadAttention, LinearPositionEmbeddingSine


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=64):
        super().__init__()
        self.patch_size = 8
        self.dim        = embed_dim
        self.proj       = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//4, kernel_size=6, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=6, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=6, stride=2, padding=2),
        )

        self.ffn_with_coord = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim*2, embed_dim*2, kernel_size=1)
        )
        self.norm = nn.LayerNorm(embed_dim*2)

    def forward(self, x) -> tuple[torch.Tensor, list[int]]:
        B, C, H, W = x.shape    # C == 1

        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        x = self.proj(x)
        out_size = x.shape[2:] 

        patch_coord = coords_grid(
            B, out_size[0], out_size[1], x.device, x.dtype
        ) * self.patch_size + (self.patch_size / 2) # in feature coordinate space
        patch_coord = patch_coord.view(B, 2, -1).permute(0, 2, 1)
        patch_coord_enc = LinearPositionEmbeddingSine(patch_coord, dim=self.dim)
        
        patch_coord_enc = patch_coord_enc.permute(0, 2, 1).view(B, -1, out_size[0], out_size[1])

        x_pe = torch.cat([x, patch_coord_enc], dim=1)
        x = self.ffn_with_coord(x_pe)
        x = self.norm(x.flatten(2).transpose(1, 2))

        return x, out_size


class VerticalSelfAttentionLayer(nn.Module):
    def __init__(self, dim, cfg, num_heads=8, dropout=0.):
        super(VerticalSelfAttentionLayer, self).__init__()
        self.cfg = cfg
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        embed_dim = dim
        mlp_ratio = 4
        ws = 7
        sr_ratio = 4
        drop_rate = dropout
        attn_drop_rate=0.

        self.local_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, sr_ratio=sr_ratio, ws=ws, vert_c_dim=cfg.vert_c_dim
        )
        self.global_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate,
            attn_drop=attn_drop_rate, sr_ratio=sr_ratio, ws=1, vert_c_dim=cfg.vert_c_dim
        )

    def forward(self, x: torch.Tensor, size: tuple[int, int], context=None):
        x = self.local_block(x, size, context)
        x = self.global_block(x, size, context)

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, proj_drop=0., dropout=0.):
        super(SelfAttentionLayer, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.multi_head_attn = MultiHeadAttention(dim, num_heads)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = x
        x = self.norm1(x)

        q, k, v = self.q(x), self.k(x), self.v(x)

        x = self.multi_head_attn(q, k, v)

        x = self.proj(x)
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x

    def compute_params(self):
        num = 0
        for param in self.parameters():
            num +=  np.prod(param.size())

        return num


class CrossAttentionLayer(nn.Module):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, num_heads=8, proj_drop=0., dropout=0.):
        super(CrossAttentionLayer, self).__init__()
        assert qk_dim % num_heads == 0, f"dim {qk_dim} should be divided by num_heads {num_heads}."
        assert v_dim % num_heads == 0, f"dim {v_dim} should be divided by num_heads {num_heads}."
        """
            Query Token:    [N, C]  -> [N, qk_dim]  (Q)
            Target Token:   [M, D]  -> [M, qk_dim]  (K),    [M, v_dim]  (V)
        """
        self.num_heads = num_heads
        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = BroadMultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, tgt_token):
        """
            x: [BH1W1, H3W3, D]
        """
        short_cut = query
        query = self.norm1(query)
        q, k, v = self.q(query), self.k(tgt_token), self.v(tgt_token)
        x = self.multi_head_attn(q, k, v)
        x = short_cut + self.proj_drop(self.proj(x))
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class CostPerceiverEncoder(nn.Module):
    def __init__(self, cfg):
        super(CostPerceiverEncoder, self).__init__()
        self.cfg = cfg
        self.patch_size: int = 8
        self.cost_heads_num: int = cfg.cost_heads_num
        self.cost_latent_token_num: int = cfg.cost_latent_token_num
        
        self.patch_embed = PatchEmbed(in_chans=self.cfg.cost_heads_num, embed_dim=cfg.cost_latent_input_dim)

        self.depth = cfg.encoder_depth

        self.latent_tokens = nn.Parameter(torch.randn(1, cfg.cost_latent_token_num, cfg.cost_latent_dim))

        query_token_dim, tgt_token_dim = cfg.cost_latent_dim, cfg.cost_latent_input_dim*2
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.input_layer = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, dropout=cfg.dropout)
        
        self.encoder_layers = nn.ModuleList([
            SelfAttentionLayer(cfg.cost_latent_dim, dropout=cfg.dropout)
            for idx in range(self.depth)
        ])
        self.vertical_encoder_layers = nn.ModuleList([
            VerticalSelfAttentionLayer(cfg.cost_latent_dim, cfg, dropout=cfg.dropout)
            for idx in range(self.depth)
        ])

    def forward(self, cost_volume: torch.Tensor, context=None) -> tuple[torch.Tensor, torch.Tensor]:
        B, heads, H1, W1, H2, W2 = cost_volume.shape
        cost_maps = cost_volume.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*H1*W1, self.cost_heads_num, H2, W2)
        
        with torch.cuda.nvtx.range("Patch Embed"):
            x, size = self.patch_embed(cost_maps)   # B*H1*W1, size[0]*size[1], C

        with torch.cuda.nvtx.range("Cross Attention"):
            x = self.input_layer(self.latent_tokens, x)

        short_cut = x
        use_ckpt = torch.is_grad_enabled()

        for layer, vert_layer in zip(self.encoder_layers, self.vertical_encoder_layers):
            
            with torch.cuda.nvtx.range("SelfAttentionLayer"):
                x = grad_checkpoint(layer, x, use_reentrant=False) if use_ckpt else layer(x)
                x = x.view(B, H1*W1, self.cost_latent_token_num, -1).permute(0, 2, 1, 3).reshape(B*self.cost_latent_token_num, H1*W1, -1)
            
            with torch.cuda.nvtx.range("Vertical SelfAttnLayer"):
                x = grad_checkpoint(vert_layer, x, (H1, W1), context, use_reentrant=False) if use_ckpt else vert_layer(x, (H1, W1), context)
                x = x.view(B, self.cost_latent_token_num, H1*W1, -1).permute(0, 2, 1, 3).reshape(B*H1*W1, self.cost_latent_token_num, -1)

        x = x + short_cut
        return x, cost_maps


class MemoryEncoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryEncoder, self).__init__()
        self.cfg = cfg

        self.feat_encoder = TwinsSVTLarge(pretrained=self.cfg.pretrain)
        self.channel_convertor = nn.Conv2d(cfg.encoder_latent_dim, cfg.encoder_latent_dim, 1, padding=0, bias=False)
        self.cost_perceiver_encoder = CostPerceiverEncoder(cfg)

    def corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        heads = self.cfg.cost_heads_num
        d = dim // heads  # each head's channel dim
        
        # Simplified feature map reshaping - combine operations
        fmap1 = fmap1.reshape(batch, heads, d, ht*wd).permute(0, 1, 3, 2)  # [b, heads, (ht*wd), d]
        fmap2 = fmap2.reshape(batch, heads, d, ht*wd).permute(0, 1, 3, 2)  # [b, heads, (ht*wd), d]
        
        # Reshape for bmm (keeping the efficient batch matrix multiplication)
        fmap1_bmm = fmap1.reshape(batch*heads, ht*wd, d)
        fmap2_bmm = fmap2.reshape(batch*heads, ht*wd, d)
        
        # Use bmm as in the original code
        corr_bmm = torch.bmm(fmap1_bmm, fmap2_bmm.transpose(1, 2))  # [b*heads, (ht*wd), (ht*wd)]
        
        # Simplified reshaping to final 6D format - direct reshape from bmm result
        corr = corr_bmm.view(batch, heads, ht*wd, ht*wd).reshape(batch, heads, ht, wd, ht, wd)
        
        return corr

    def _corr_and_encode(self, feat_s: torch.Tensor, feat_t: torch.Tensor,
                         context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 6-D cost volume and encode it.

        Factored out so it can be wrapped with gradient checkpointing — the
        massive [B, heads, H, W, H, W] cost volume becomes a recomputed
        intermediate instead of being held for backward.
        """
        cost_volume = self.corr(feat_s, feat_t).to(feat_s)
        return self.cost_perceiver_encoder(cost_volume, context)

    def forward(self, img1, img2, context):
        imgs = torch.cat([img1, img2], dim=0)
        
        with torch.cuda.nvtx.range("Feature Encoder"):
            feats = self.feat_encoder(imgs)
        
        with torch.cuda.nvtx.range("Channel Convertor"):
            feats = self.channel_convertor(feats)
        
        B = feats.shape[0] // 2
        feat_s, feat_t = feats[:B], feats[B:]
        
        with torch.cuda.nvtx.range("Corr Volume + Cost Perceiver Enc"):
            if torch.is_grad_enabled():
                x, cost_maps = grad_checkpoint(
                    self._corr_and_encode, feat_s, feat_t, context,
                    use_reentrant=False,
                )
            else:
                x, cost_maps = self._corr_and_encode(feat_s, feat_t, context)

        return x, cost_maps
