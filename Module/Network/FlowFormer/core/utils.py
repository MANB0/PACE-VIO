import torch
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        if sum(self._pad) == 0: return inputs
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def bilinear_sampler(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """ Wrapper for grid_sample, uses pixel coordinates.
    
    cuDNN grid_sample backward does not support bf16 / non-contiguous layouts,
    so cuDNN is disabled here to fall back to the native implementation.
    """
    H, W = img.shape[-2:]

    coords = coords.clone()
    coords[..., 0] = 2 * coords[..., 0] / (W-1) - 1
    coords[..., 1] = 2 * coords[..., 1] / (H-1) - 1

    # Defact in CUDNN (Apr 2026) Ref: https://github.com/pytorch/pytorch/issues/88380
    with torch.backends.cudnn.flags(enabled=False):
        return F.grid_sample(img, coords, align_corners=True)

def coords_grid(batch: int, ht: int, wd: int, device: torch.device, dtype: torch.dtype):
    coords = torch.meshgrid(
        torch.arange(0, ht, device=device, dtype=dtype),
        torch.arange(0, wd, device=device, dtype=dtype),
        indexing="ij"
    )
    coords = torch.stack((coords[1], coords[0]), dim=0)
    return coords.unsqueeze(0).repeat(batch, 1, 1, 1)
