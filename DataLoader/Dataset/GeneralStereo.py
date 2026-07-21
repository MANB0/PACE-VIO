import cv2
import torch
import numpy as np
import pypose as pp
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from torch.utils.data import Dataset

from ..Interface import StereoFrame, StereoData
from ..SequenceBase import SequenceBase


class GeneralStereoSequence(SequenceBase[StereoFrame]):
    @classmethod
    def name(cls) -> str: return "GeneralStereo"
    
    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        
        # metadata
        self.seqRoot = Path(cfg.root)
        self.baseline = cfg.bl
        self.T_BS = pp.identity_SE3(1, dtype=torch.float64)
        default_pose_frame = "NWU" if Path(self.seqRoot, "ref_pose.csv").exists() else "NED"
        self.pose_output_frame = str(getattr(cfg, "pose_output_frame", default_pose_frame)).upper()
        
        self.ImageL    = MonocularDataset(Path(self.seqRoot, "left"), cfg.format)
        self.ImageR    = MonocularDataset(Path(self.seqRoot, "right"), cfg.format)
        assert len(self.ImageL) == len(self.ImageR)
        assert self.ImageL.timestamps == self.ImageR.timestamps, "Left/right image timestamps are not aligned"
        
        if hasattr(cfg.camera, "fx"):
            self.K = torch.tensor([[
                [cfg.camera.fx, 0., cfg.camera.cx], 
                [0., cfg.camera.fy, cfg.camera.cy],
                [0.           , 0., 1.           ]
            ]], dtype=torch.float).repeat(len(self.ImageL), 1, 1)
        else:
            self.K = torch.tensor(np.load(Path(self.seqRoot, "intrinsic.npy"))).float()

        self.length = len(self.ImageL)
        super().__init__(self.length)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        imageL = self.ImageL[index]
        imageR = self.ImageR[index]
        frame_ns = self.ImageL.timestamps[index]
            
        return StereoFrame(
            idx    = [local_index],
            time_ns= [frame_ns],
            stereo = StereoData(
                T_BS     = self.T_BS,
                K        = self.K[index:index+1],
                baseline = torch.tensor([self.baseline]),
                width    = imageL.size(-1),
                height   = imageL.size(-2),
                time_ns  = [frame_ns],
                imageL   = imageL,
                imageR   = imageR
            )
        )

    @classmethod
    def is_valid_config(cls, config) -> None:
        cls._enforce_config_spec(config, {
            "root"  : lambda s: isinstance(s, str),
            "bl"    : lambda v: isinstance(v, float),
            "format": lambda s: isinstance(s, str),
            "camera": lambda v: isinstance(v, dict) and (len(v) == 0 or cls._enforce_config_spec(v, {
                "fx": lambda v: isinstance(v, float),
                "fy": lambda v: isinstance(v, float),
                "cx": lambda v: isinstance(v, float),
                "cy": lambda v: isinstance(v, float)
            }, allow_excessive_cfg=True)) or True
        }, allow_excessive_cfg=True)

class MonocularDataset(Dataset):
    """
    Return images in the given directory ends with .png
    Return the image in shape (1, 3, H, W) with dtype=float32 
    and normalized (image in [0, 1])
    """
    def __init__(self, directory: Path, format: Literal["png", "jpg"]) -> None:
        super().__init__()
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory {self.directory} does not exist"
        
        # Sort numerically by timestamp (stem as int) to handle inconsistent digit counts in filenames
        self.file_names = sorted(directory.glob(f"*.{format}"), key=lambda p: int(p.stem))
            
        self.length = len(self.file_names)
        assert self.length > 0, f"No file with '.png' suffix is found under {self.directory}"
        self.timestamps = [self._timestamp_from_name(path, idx) for idx, path in enumerate(self.file_names)]

    @staticmethod
    def _timestamp_from_name(path: Path, fallback_index: int) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return fallback_index * 1000

    @staticmethod
    def load_png_format(path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Failed to read image from {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Output image tensor in shape of (1, C, H, W)
        image = self.load_png_format(str(self.file_names[index]))
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.
        return image
