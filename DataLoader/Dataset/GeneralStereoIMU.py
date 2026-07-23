import cv2
import torch
import numpy as np
import pypose as pp
import json
import re
import yaml

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..Interface import StereoFrame, StereoInertialFrame, StereoData, IMUData
from ..SequenceBase import SequenceBase
from Utility.IMUKinematics import format_imu_sigma, is_valid_imu_sigma
from Utility.IMUCSV import IMUCSVLoader
from Utility.PrettyPrint import Logger


def _load_t_ci(extrinsics: dict | None) -> pp.LieTensor:
    """Load the sole camera/IMU extrinsic, p_C = T_CI p_I."""
    if not isinstance(extrinsics, dict) or "T_CI" not in extrinsics:
        raise ValueError(
            "metadata.extrinsics.T_CI is required and must be a 4x4 transform "
            "mapping the raw IMU measurement frame I to the MACVO camera frame C"
        )
    unexpected = sorted(set(extrinsics) - {"T_CI"})
    if unexpected:
        raise ValueError(
            "metadata.extrinsics must contain only T_CI; remove legacy or duplicate "
            f"extrinsics: {unexpected}"
        )

    matrix = torch.as_tensor(extrinsics["T_CI"], dtype=torch.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"metadata.extrinsics.T_CI must have shape 4x4, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("metadata.extrinsics.T_CI contains NaN or Inf")
    if not torch.allclose(
        matrix[3],
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=matrix.dtype),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("metadata.extrinsics.T_CI must have homogeneous bottom row [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    identity = torch.eye(3, dtype=matrix.dtype)
    if not torch.allclose(rotation.T @ rotation, identity, atol=1e-7, rtol=1e-7):
        raise ValueError("metadata.extrinsics.T_CI rotation is not orthonormal")
    determinant = float(torch.linalg.det(rotation).item())
    if abs(determinant - 1.0) > 1e-7:
        raise ValueError(
            "metadata.extrinsics.T_CI rotation must be proper (det=+1), "
            f"got det={determinant:.9g}"
        )
    return pp.from_matrix(matrix.unsqueeze(0), pp.SE3_type)


def _imu_calibration_axis_value(imu_calib: dict, key: str):
    """Read an isotropic sigma or an optional x/y/z sigma triplet."""
    xyz_key = f"{key}XYZ"
    if xyz_key in imu_calib:
        return imu_calib[xyz_key]
    value = imu_calib[key]
    if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
        return [value["x"], value["y"], value["z"]]
    return value


def _continuous_imu_noise_density(imu_calib: dict, key: str):
    """Read one continuous-time density as an isotropic or three-axis value."""
    value = _imu_calibration_axis_value(imu_calib, key)
    if not is_valid_imu_sigma(value):
        raise ValueError(
            f"imu.{key} must be a finite non-negative scalar or three-axis density"
        )
    values = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
    if values.numel() == 1:
        return float(values.item())
    return tuple(float(axis) for axis in values.tolist())


def _metadata_gravity_m_s2(meta: dict | None) -> float | None:
    imu_meta = meta.get("imu", {}) if meta else {}
    gravity_m_s2 = imu_meta.get("gravity_m_s2", None)
    return None if gravity_m_s2 is None else float(gravity_m_s2)


class MonocularDataset:
    """
    Return images in shape (1, 3, H, W), float32, normalized to [0, 1].
    """

    def __init__(self, directory: Path, image_format: str) -> None:
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory does not exist: {self.directory}"

        # Sort numerically by timestamp (stem as int) to handle inconsistent digit counts in filenames
        self.file_names = sorted(directory.glob(f"*.{image_format}"), key=lambda p: int(p.stem))
        assert len(self.file_names) > 0, f"No image with '.{image_format}' suffix under {self.directory}"

        self.timestamps = [int(p.stem) for p in self.file_names]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {self.file_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.0
        return image


class GeneralStereoIMUSequence(SequenceBase[StereoInertialFrame]):
    @classmethod
    def name(cls) -> str:
        return "GeneralStereoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)

        self.seq_root = Path(cfg.root)
        default_pose_frame = "NWU" if Path(self.seq_root, "ref_pose.csv").exists() else "NED"
        self.pose_output_frame = str(getattr(cfg, "pose_output_frame", default_pose_frame)).upper()
        self.gravity = float(getattr(cfg, "gravity", 9.81))
        self.gravity_source = "config"
        self.imu_window_ns = int(getattr(cfg, "imu_window_ns", 100_000_000))
        self.imu_fallback_max_dt_ns = int(getattr(cfg, "imu_fallback_max_dt_ns", 50_000_000))

        self.cam_T_BS = pp.identity_SE3(1, dtype=torch.float64)
        self.imu_vio_sensor_T_imu = pp.identity_SE3(1, dtype=torch.float64)

        self.image_l = MonocularDataset(Path(self.seq_root, "left"), cfg.format)
        self.image_r = MonocularDataset(Path(self.seq_root, "right"), cfg.format)
        assert len(self.image_l) == len(self.image_r), "Left and right images are not same length"
        assert self.image_l.timestamps == self.image_r.timestamps, "Left/right timestamps are not aligned"

        # ── Load metadata.json (new format) once for all params ─────────
        meta: dict | None = None
        meta_json_path = Path(self.seq_root, "metadata.json")
        if meta_json_path.exists():
            with open(meta_json_path) as f:
                meta = json.load(f)
            Logger.write("info", f"Loaded metadata.json from {self.seq_root.name}")

        # ── Camera intrinsics ───────────────────────────────────────────
        cam_meta = meta.get("camera", None) if meta else None
        if cam_meta is None:
            cam_meta_path = Path(self.seq_root, "camera_metadata.json")
            if cam_meta_path.exists():
                with open(cam_meta_path) as f:
                    cam_meta = json.load(f)
                Logger.write("info", f"Camera params from camera_metadata.json (fallback)")

        if cam_meta is not None:
            self.K = torch.tensor([[[cam_meta["fx"], 0., cam_meta["cx"]],
                                     [0., cam_meta["fy"], cam_meta["cy"]],
                                     [0., 0., 1.]]], dtype=torch.float32)
            self.baseline = float(cam_meta.get("baseline_m", 0.225))
            Logger.write("info", f"Camera: K,fx={cam_meta['fx']},fy={cam_meta['fy']}, bl={self.baseline}m")
        else:
            if hasattr(cfg, "camera") and hasattr(cfg.camera, "fx"):
                self.K = torch.tensor([[[cfg.camera.fx, 0., cfg.camera.cx],
                                         [0., cfg.camera.fy, cfg.camera.cy],
                                         [0., 0., 1.]]], dtype=torch.float32)
            else:
                self.K = torch.tensor(np.load(Path(self.seq_root, "intrinsic.npy")), dtype=torch.float32)
            self.baseline = float(getattr(cfg, "bl", 0.225))
            Logger.write("info", "No camera_metadata.json, using config/npy defaults")
        # Raw IMU samples remain in their CSV frame throughout initialization and
        # preintegration. T_CI is used only at camera/IMU pose boundaries.
        self.imu_source_world_frame = "NWU"
        self.imu_source_measurement_frame = "FLU"
        self.imu_internal_world_frame = "NED"
        self.imu_internal_measurement_frame = "FLU"
        self.imu_world_frame = "NED"
        self.imu_measurement_frame = "FLU"
        self.imu_metadata_gravity_m_s2 = _metadata_gravity_m_s2(meta)
        if self.imu_metadata_gravity_m_s2 is not None:
            self.gravity = float(self.imu_metadata_gravity_m_s2)
            self.gravity_source = "metadata.json"
        self.imu_T_BS = pp.identity_SE3(1, dtype=torch.float64)
        self.imu_world_frame = self.imu_internal_world_frame
        Logger.write("info", "Raw IMU samples remain in FLU; no hidden FLU/NED pre-rotation is applied")

        # ── IMU calibration & extrinsic ─────────────────────────────────
        self.imu_calib_acc_sigma = None
        self.imu_calib_gyro_sigma = None
        self.imu_calib_acc_w_sigma = None
        self.imu_calib_gyro_w_sigma = None
        self.imu_calib_source = None

        imu_calib = meta.get("imu", None) if meta else None
        imu_extrinsic = meta.get("extrinsics", None) if meta else None
        if imu_calib is not None:
            self.imu_calib_source = "metadata.json"
        if imu_calib is None:
            calib_path = Path(self.seq_root, "imu_calibration.yaml")
            if calib_path.exists():
                try:
                    raw = calib_path.read_text(encoding="utf-8")
                    raw = re.sub(r'^%YAML:.*\n', '', raw)
                    imu_calib = yaml.safe_load(raw)
                    self.imu_calib_source = "imu_calibration.yaml"
                    Logger.write("info", "IMU params from imu_calibration.yaml (fallback)")
                except Exception as e:
                    Logger.write("warn", f"Failed to parse imu_calibration.yaml: {e}")

        if imu_calib is not None:
            required_noise_fields = {"NoiseAcc", "NoiseGyro", "AccWalk", "GyroWalk"}
            missing_noise_fields = sorted(required_noise_fields.difference(imu_calib))
            if missing_noise_fields:
                raise ValueError(
                    "IMU metadata must provide continuous-time densities "
                    f"{sorted(required_noise_fields)}; missing {missing_noise_fields}"
                )
            self.imu_calib_acc_sigma = _continuous_imu_noise_density(imu_calib, "NoiseAcc")
            self.imu_calib_gyro_sigma = _continuous_imu_noise_density(imu_calib, "NoiseGyro")
            self.imu_calib_acc_w_sigma = _continuous_imu_noise_density(imu_calib, "AccWalk")
            self.imu_calib_gyro_w_sigma = _continuous_imu_noise_density(imu_calib, "GyroWalk")
            Logger.write("info",
                f"IMU calib: sigma_a={format_imu_sigma(self.imu_calib_acc_sigma)}, "
                f"sigma_g={format_imu_sigma(self.imu_calib_gyro_sigma)}, "
                f"sigma_aw={format_imu_sigma(self.imu_calib_acc_w_sigma)}, "
                f"sigma_gw={format_imu_sigma(self.imu_calib_gyro_w_sigma)}, "
                "continuous-time densities consumed directly")

        self.imu_vio_sensor_T_imu = _load_t_ci(imu_extrinsic)
        Logger.write(
            "info",
            "Loaded metadata.extrinsics.T_CI with contract p_C = T_CI p_I "
            "(raw IMU FLU frame I to MACVO camera frame C)",
        )

        self.frame_timestamps = self.image_l.timestamps

        # ── IMU data: prefer imu_data.csv, fallback to imu.csv ──────────
        imu_csv_path = Path(self.seq_root, "imu_data.csv")
        if not imu_csv_path.exists():
            imu_csv_legacy = Path(self.seq_root, "imu.csv")
            if imu_csv_legacy.exists():
                Logger.write("warn", f"imu_data.csv not found, falling back to imu.csv")
                imu_csv_path = imu_csv_legacy
            else:
                raise FileNotFoundError(f"Neither imu_data.csv nor imu.csv found in {self.seq_root}")
        self.imu_loader = IMUCSVLoader(imu_csv_path)

        # ── Metadata usage summary ────────────────────────────────────
        _log_metadata_usage(self.seq_root, meta, cam_meta, imu_calib, imu_extrinsic)

        Logger.write(
            "info",
            "GeneralStereoIMU timing contract: timestamps are nanoseconds and "
            "camera/IMU offset is fixed to zero",
        )

        super().__init__(len(self.image_l))

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        frame_ns = self.frame_timestamps[index]
        prev_ns = self.frame_timestamps[index - 1] if index > 0 else (frame_ns - self.imu_window_ns)

        imu_time_ns, imu_acc, imu_gyro, imu_sampling_map = (
            self.imu_loader.query_range_with_sampling_map(prev_ns, frame_ns)
        )
        if imu_time_ns.numel() == 0:
            _, nearest_time_ns, nearest_acc, nearest_gyro = self.imu_loader.query_nearest(frame_ns)
            dt_ns = abs(int(nearest_time_ns[0].item()) - frame_ns)

            use_nearest = (self.imu_fallback_max_dt_ns < 0) or (dt_ns <= self.imu_fallback_max_dt_ns)
            if use_nearest:
                imu_time_ns, imu_acc, imu_gyro = nearest_time_ns, nearest_acc, nearest_gyro
            else:
                imu_time_ns = torch.empty((0,), dtype=torch.long)
                imu_acc = torch.empty((0, 3), dtype=torch.float32)
                imu_gyro = torch.empty((0, 3), dtype=torch.float32)
            imu_sampling_map = None

        image_l = self.image_l[index]
        image_r = self.image_r[index]

        frame = StereoInertialFrame(
            idx=[local_index],
            time_ns=[frame_ns],
            stereo=StereoData(
                T_BS=self.cam_T_BS,
                K=self.K,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                width=image_l.size(-1),
                height=image_l.size(-2),
                time_ns=[frame_ns],
                imageL=image_l,
                imageR=image_r,
            ),
            imu=IMUData(
                T_BS=self.imu_T_BS,
                time_ns=imu_time_ns.view(1, -1, 1),
                gravity=[self.gravity],
                acc=imu_acc.view(1, -1, 3),
                gyro=imu_gyro.view(1, -1, 3),
            ),
        )
        # Attach IMU calibration noise densities (used by MACVO._estimate_imu_priors)
        if self.imu_calib_acc_sigma is not None:
            frame.imu_calib_acc_sigma = self.imu_calib_acc_sigma
            frame.imu_calib_gyro_sigma = self.imu_calib_gyro_sigma
            frame.imu_calib_source = self.imu_calib_source
        if imu_sampling_map is not None:
            frame.imu_sampling_raw_time_ns = imu_sampling_map.raw_time_ns
            frame.imu_sampling_knot_from_raw = imu_sampling_map.knot_from_raw
        if self.imu_calib_acc_w_sigma is not None:
            frame.imu_calib_acc_w_sigma = self.imu_calib_acc_w_sigma
            frame.imu_calib_gyro_w_sigma = self.imu_calib_gyro_w_sigma
        frame.imu_source_world_frame = self.imu_source_world_frame
        frame.imu_source_measurement_frame = self.imu_source_measurement_frame
        frame.imu_internal_world_frame = self.imu_internal_world_frame
        frame.imu_internal_measurement_frame = self.imu_internal_measurement_frame
        frame.imu_world_frame = self.imu_world_frame
        frame.imu_measurement_frame = self.imu_measurement_frame
        frame.imu_metadata_gravity_m_s2 = self.imu_metadata_gravity_m_s2
        frame.imu_gravity_source = self.gravity_source
        frame.imu_vio_sensor_T_imu = self.imu_vio_sensor_T_imu

        return frame

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        def _camera_valid(value: object) -> bool:
            if isinstance(value, dict):
                if len(value) == 0:
                    return True
                camera_cfg = SimpleNamespace(**value)
            elif isinstance(value, SimpleNamespace):
                camera_cfg = value
            else:
                return False
            try:
                cls._enforce_config_spec(camera_cfg, {
                    "fx": lambda x: isinstance(x, (float, int)),
                    "fy": lambda x: isinstance(x, (float, int)),
                    "cx": lambda x: isinstance(x, (float, int)),
                    "cy": lambda x: isinstance(x, (float, int)),
                }, allow_excessive_cfg=True)
            except Exception:
                return False
            return True

        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "bl": lambda v: isinstance(v, (float, int)) and v > 0,
            "format": lambda s: isinstance(s, str),
            "gravity": lambda v: isinstance(v, (float, int)) and v > 0,
        }, allow_excessive_cfg=True)
        optional_spec = {
            "imu_window_ns": lambda v: isinstance(v, int) and v > 0,
            "imu_fallback_max_dt_ns": lambda v: isinstance(v, int) and v >= -1,
            "camera": _camera_valid,
        }
        for key, test_fn in optional_spec.items():
            if key in config.__dict__ and not test_fn(config.__dict__[key]):
                raise ValueError(f"Config does not match specification! ({key}={config.__dict__[key]!r})")


def _log_metadata_usage(seq_root, meta, cam_meta, imu_calib, imu_extrinsic):
    """Print a concise Metadata Usage Summary for downstream audit."""
    from pathlib import Path as _Path
    _ = _Path
    root_name = _Path(seq_root).name
    lines = [f"── Metadata Usage Summary [{root_name}] ──"]

    if cam_meta:
        lines.append(f"  camera: fx={cam_meta.get('fx')}, fy={cam_meta.get('fy')}, "
                      f"bl={cam_meta.get('baseline_m')}m "
                      f"(from {'metadata.json' if meta and 'camera' in meta else 'fallback'})")
    else:
        lines.append("  camera: from config defaults (no metadata)")

    if imu_calib:
        src = "metadata.json" if meta and "imu" in meta else "imu_calibration.yaml fallback"
        lines.append(f"  imu continuous densities: sigma_a={imu_calib.get('NoiseAcc')}, "
                      f"sigma_g={imu_calib.get('NoiseGyro')}, "
                      f"sigma_aw={imu_calib.get('AccWalk')}, "
                      f"sigma_gw={imu_calib.get('GyroWalk')} (from {src})")
    else:
        lines.append("  imu noise: from config defaults (no calibration)")

    if imu_extrinsic:
        lines.append("  extrinsic: sole 4x4 T_CI loaded with p_C = T_CI p_I")
        lines.append(f"  T_CI={imu_extrinsic.get('T_CI', 'MISSING')}")
        lines.append("  raw IMU samples: retained in frame I; T_CI is not applied during preintegration")
    else:
        lines.append("  extrinsic: MISSING (metadata.extrinsics.T_CI is required)")

    if meta:
        imu_meta = meta.get("imu", {})
        gt_meta = meta.get("ground_truth", {})
        lines.append(f"  imu gravity magnitude={imu_meta.get('gravity_m_s2','?')}")
        lines.append("  IMU samples: source FLU retained as optimizer/preintegration frame I")
        lines.append("  world frame: internal NED; initial R_WI is supplied by static initialization/T_CI")
        lines.append(f"  GT quat: meaning={gt_meta.get('quaternion_meaning','?')}, "
                     f"order={gt_meta.get('quaternion_order','?')}")
        lines.append(f"  GT pos frame: {gt_meta.get('position_frame','?')}")
        lines.append(
            "  GT velocity frame: "
            f"{gt_meta.get('velocity_frame', 'HoloOcean ref_pose world NWU') }"
        )
        lines.append("  timing: nanosecond timestamps, fixed zero camera/IMU offset (runtime contract)")

    for line in lines:
        Logger.write("info", line)
