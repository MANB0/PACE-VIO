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
from Utility.IMUKinematics import (
    format_imu_sigma,
    imu_bias_sigma_to_continuous_random_walk_density,
    imu_sigma_to_continuous_density,
)
from Utility.IMUCSV import IMUCSVLoader
from Utility.PrettyPrint import Logger


def _flu_to_ned_se3() -> pp.LieTensor:
    # PyPose SE3 stores quaternions as qx, qy, qz, qw.
    return pp.SE3(torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    ).reshape(1, 7))


def _body_nwu_translation_to_internal_ned(translation: list[float]) -> list[float]:
    return [float(translation[0]), -float(translation[1]), -float(translation[2])]


def _translation_se3(translation: list[float]) -> pp.LieTensor:
    return pp.SE3(torch.tensor(
        [[float(translation[0]), float(translation[1]), float(translation[2]), 0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ))


def _imu_calibration_axis_value(imu_calib: dict, key: str):
    """Read an isotropic sigma or an optional x/y/z sigma triplet."""
    xyz_key = f"{key}XYZ"
    if xyz_key in imu_calib:
        return imu_calib[xyz_key]
    value = imu_calib[key]
    if isinstance(value, dict) and all(axis in value for axis in ("x", "y", "z")):
        return [value["x"], value["y"], value["z"]]
    return value


def _metadata_uses_holoocean_nwu_flu(meta: dict | None) -> bool:
    if not meta:
        return False
    coord = meta.get("coordinate_convention", {})
    imu_meta = meta.get("imu", {})
    world_frame = str(coord.get("export_world_frame", coord.get("holocean_world_frame", ""))).upper()
    body_frame = str(coord.get("body_frame", "")).upper()
    camera_frame = str(coord.get("camera_frame", "")).upper()
    imu_frame = str(coord.get("imu_measurement_frame", imu_meta.get("frame", ""))).upper()
    return (
        "NWU" in world_frame
        and "NWU" in body_frame
        and "FLU" in imu_frame
        and ("BODY NWU" in camera_frame or "ALIGNED" in camera_frame)
    )


def _canonical_unit(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("²", "^2")


def _validate_imu_metadata_conventions(meta: dict | None) -> dict[str, Any]:
    imu_meta = meta.get("imu", {}) if meta else {}
    ts_meta = meta.get("time_synchronization", {}) if meta else {}
    dataset_meta = meta.get("dataset", {}) if meta else {}

    acc_unit = imu_meta.get("acc_unit", "m/s^2")
    gyro_unit = imu_meta.get("gyro_unit", "rad/s")
    timestamp_unit = ts_meta.get("timestamp_unit", dataset_meta.get("timestamp_unit", "ns"))
    acc_includes_gravity = imu_meta.get("acc_includes_gravity", True)
    gravity_m_s2 = imu_meta.get("gravity_m_s2", None)

    if _canonical_unit(acc_unit) not in {"m/s^2", "m/s2"}:
        raise ValueError(f"Unsupported IMU acc_unit={acc_unit!r}; expected m/s^2")
    if _canonical_unit(gyro_unit) != "rad/s":
        raise ValueError(f"Unsupported IMU gyro_unit={gyro_unit!r}; expected rad/s")
    if _canonical_unit(timestamp_unit) != "ns":
        raise ValueError(f"Unsupported timestamp_unit={timestamp_unit!r}; expected ns")
    if not bool(acc_includes_gravity):
        raise ValueError("Unsupported imu.acc_includes_gravity=False; current preintegration expects gravity-inclusive acceleration")

    gravity_value = None if gravity_m_s2 is None else float(gravity_m_s2)
    return {
        "acc_unit": "m/s^2",
        "gyro_unit": "rad/s",
        "timestamp_unit": "ns",
        "acc_includes_gravity": True,
        "gravity_m_s2": gravity_value,
    }


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
        self.auto_estimate_time_offset = bool(getattr(cfg, "auto_estimate_time_offset", True))
        self.imu_time_offset_ns = int(getattr(cfg, "imu_time_offset_ns", 0))
        self.imu_time_offset_source = "config"

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
        # ── imu_T_BS: body-to-sensor extrinsic; IMU samples use inverse rotation
        #    when they are converted back into the optimizer body frame.
        self.imu_source_world_frame = "NED"
        self.imu_source_measurement_frame = "FLU"
        self.imu_internal_world_frame = "NED"
        self.imu_internal_measurement_frame = "NED"
        self.imu_world_frame = "NED"
        self.imu_measurement_frame = "FLU"
        self.imu_acc_unit = "m/s^2"
        self.imu_gyro_unit = "rad/s"
        self.imu_timestamp_unit = "ns"
        self.imu_metadata_gravity_m_s2 = None
        self.imu_acc_includes_gravity = True
        imu_conventions = _validate_imu_metadata_conventions(meta)
        self.imu_acc_unit = imu_conventions["acc_unit"]
        self.imu_gyro_unit = imu_conventions["gyro_unit"]
        self.imu_timestamp_unit = imu_conventions["timestamp_unit"]
        self.imu_acc_includes_gravity = imu_conventions["acc_includes_gravity"]
        self.imu_metadata_gravity_m_s2 = imu_conventions["gravity_m_s2"]
        if self.imu_metadata_gravity_m_s2 is not None:
            self.gravity = float(self.imu_metadata_gravity_m_s2)
            self.gravity_source = "metadata.json"
        if _metadata_uses_holoocean_nwu_flu(meta):
            self.imu_T_BS = _flu_to_ned_se3()
            self.imu_source_world_frame = "NWU"
            self.imu_source_measurement_frame = "FLU"
            self.imu_internal_world_frame = "NED"
            self.imu_internal_measurement_frame = "NED"
            self.imu_world_frame = self.imu_internal_world_frame
            Logger.write("info", "imu_T_BS rotation is set from metadata-declared HoloOcean FLU/NWU to MACVO internal NED convention")
        else:
            # Legacy path: FLU (x-fwd,y-left,z-up) → NED (x-fwd,y-right,z-down).
            self.imu_T_BS = _flu_to_ned_se3()
            Logger.write("info", "imu_T_BS rotation set to R_x(180°) for legacy FLU/NED convention")

        # ── IMU calibration & extrinsic ─────────────────────────────────
        self.imu_calib_acc_sigma = None
        self.imu_calib_gyro_sigma = None
        self.imu_calib_acc_w_sigma = None
        self.imu_calib_gyro_w_sigma = None
        self.imu_calib_sigma_unit = None
        self.imu_calib_bias_sigma_unit = None
        self.imu_calib_source = None
        self.imu_calib_measurement_rate_hz = None
        self.imu_calib_bias_random_walk_update_hz = None

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

        if imu_calib is not None and "AccelSigma" in imu_calib and "AngVelSigma" in imu_calib:
            rate_hz = float(imu_calib.get("rate_hz", 100))
            bias_rate_hz = float(imu_calib.get("bias_random_walk_update_hz", rate_hz))
            sigma_unit = imu_calib.get("sigma_unit", "legacy_sqrt_rate_scaled")
            bias_sigma_unit = imu_calib.get("bias_sigma_unit", sigma_unit)
            self.imu_calib_measurement_rate_hz = rate_hz
            self.imu_calib_bias_random_walk_update_hz = bias_rate_hz
            self.imu_calib_sigma_unit = str(sigma_unit)
            self.imu_calib_bias_sigma_unit = str(bias_sigma_unit)
            self.imu_calib_acc_sigma = imu_sigma_to_continuous_density(
                _imu_calibration_axis_value(imu_calib, "AccelSigma"), rate_hz, sigma_unit
            )
            self.imu_calib_gyro_sigma = imu_sigma_to_continuous_density(
                _imu_calibration_axis_value(imu_calib, "AngVelSigma"), rate_hz, sigma_unit
            )
            if "AccelBiasSigma" in imu_calib:
                self.imu_calib_acc_w_sigma = imu_bias_sigma_to_continuous_random_walk_density(
                    _imu_calibration_axis_value(imu_calib, "AccelBiasSigma"),
                    bias_rate_hz,
                    bias_sigma_unit,
                )
            if "AngVelBiasSigma" in imu_calib:
                self.imu_calib_gyro_w_sigma = imu_bias_sigma_to_continuous_random_walk_density(
                    _imu_calibration_axis_value(imu_calib, "AngVelBiasSigma"),
                    bias_rate_hz,
                    bias_sigma_unit,
                )
            Logger.write("info",
                f"IMU calib: sigma_a={format_imu_sigma(self.imu_calib_acc_sigma)}, "
                f"sigma_g={format_imu_sigma(self.imu_calib_gyro_sigma)}, "
                f"sigma_aw={format_imu_sigma(self.imu_calib_acc_w_sigma)}, "
                f"sigma_gw={format_imu_sigma(self.imu_calib_gyro_w_sigma)}, "
                f"measurement_rate_hz={rate_hz:g}, bias_update_rate_hz={bias_rate_hz:g}, "
                f"sigma_unit={self.imu_calib_sigma_unit}, bias_sigma_unit={self.imu_calib_bias_sigma_unit}")

        # IMU extrinsic → imu_T_BS translation. For HoloOcean metadata, T_body_imu
        # matches IMUData.T_BS semantics and is converted from body NWU to the
        # MACVO internal NED frame. Older fallback metadata may only expose an
        # imu-camera translation; keep that path for compatibility.
        if imu_extrinsic is not None:
            trans = None
            trans_source = ""
            sensor_imu_set = False
            if _metadata_uses_holoocean_nwu_flu(meta):
                t_body_imu = imu_extrinsic.get("T_body_imu", {})
                body_imu_trans = t_body_imu.get("translation_body_nwu_m", None)
                if body_imu_trans and len(body_imu_trans) == 3:
                    trans = _body_nwu_translation_to_internal_ned(body_imu_trans)
                    trans_source = "metadata.json T_body_imu converted NWU→NED"
                t_body_camera = imu_extrinsic.get("T_body_camera", {})
                body_camera_trans = t_body_camera.get("translation_body_nwu_m", None)
                if body_imu_trans and len(body_imu_trans) == 3 and body_camera_trans and len(body_camera_trans) == 3:
                    camera_imu_trans_nwu = [
                        float(body_imu_trans[axis]) - float(body_camera_trans[axis])
                        for axis in range(3)
                    ]
                    camera_imu_trans = _body_nwu_translation_to_internal_ned(camera_imu_trans_nwu)
                    self.imu_vio_sensor_T_imu = _translation_se3(camera_imu_trans)
                    sensor_imu_set = True
                    Logger.write(
                        "info",
                        "imu_vio_sensor_T_imu from T_body_imu - T_body_camera "
                        f"converted NWU→NED: {camera_imu_trans}",
                    )
            t_imu_cam = imu_extrinsic.get("T_imu_camera", {})
            imu_camera_trans = t_imu_cam.get("translation_body_nwu_m", None)
            if trans is None:
                trans = imu_camera_trans
                trans_source = "metadata.json T_imu_camera fallback"
            if not sensor_imu_set and imu_camera_trans and len(imu_camera_trans) == 3:
                camera_imu_trans_nwu = [-float(imu_camera_trans[axis]) for axis in range(3)]
                camera_imu_trans = _body_nwu_translation_to_internal_ned(camera_imu_trans_nwu)
                self.imu_vio_sensor_T_imu = _translation_se3(camera_imu_trans)
                Logger.write(
                    "info",
                    "imu_vio_sensor_T_imu from -T_imu_camera fallback "
                    f"converted NWU→NED: {camera_imu_trans}",
                )
            if trans and len(trans) == 3:
                tbs = self.imu_T_BS.tensor().reshape(-1).tolist()
                tbs[:3] = trans
                self.imu_T_BS = pp.SE3(torch.tensor(tbs, dtype=torch.float64).reshape(1, 7))
                Logger.write("info", f"imu_T_BS translation from {trans_source}: {trans}")
        elif imu_calib is not None and "imu_camera_extrinsic" in imu_calib:
            ext = imu_calib["imu_camera_extrinsic"]
            trans = ext.get("translation_body_nwu_m", None)
            if trans and len(trans) == 3:
                tbs = self.imu_T_BS.tensor().reshape(-1).tolist()
                tbs[:3] = trans
                self.imu_T_BS = pp.SE3(torch.tensor(tbs, dtype=torch.float64).reshape(1, 7))
                Logger.write("info", f"imu_T_BS translation from calib: {trans}")

        self.frame_timestamps = self.image_l.timestamps
        metadata_time_offset_ns = None
        if meta:
            ts_meta = meta.get("time_synchronization", {})
            for key in ("camera_imu_time_offset_ns", "imu_time_offset_ns"):
                if key in ts_meta:
                    metadata_time_offset_ns = int(ts_meta[key])
                    self.imu_time_offset_source = f"metadata.time_synchronization.{key}"
                    break

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

        if metadata_time_offset_ns is not None:
            self.imu_time_offset_ns = metadata_time_offset_ns
        elif self.auto_estimate_time_offset:
            # Only infer an offset when metadata does not provide one. Camera and
            # IMU streams can have different final timestamps solely because of
            # different sampling rates, which would create a false endpoint offset.
            cam_first, cam_last = self.frame_timestamps[0], self.frame_timestamps[-1]
            imu_first = int(self.imu_loader.time_ns[0].item())
            imu_last = int(self.imu_loader.time_ns[-1].item())

            offset_head = cam_first - imu_first
            offset_tail = cam_last - imu_last
            self.imu_time_offset_ns = int((offset_head + offset_tail) // 2)
            self.imu_time_offset_source = "auto_endpoint_average"

        # ── Metadata usage summary ────────────────────────────────────
        _log_metadata_usage(self.seq_root, meta, cam_meta, imu_calib, imu_extrinsic)

        Logger.write(
            "info",
            (
                "GeneralStereoIMU time offset (camera_ns - imu_ns) = "
                f"{self.imu_time_offset_ns} ns, "
                f"auto_estimate={self.auto_estimate_time_offset}, "
                f"source={self.imu_time_offset_source}, "
                f"fallback_max_dt_ns={self.imu_fallback_max_dt_ns}"
            ),
        )

        super().__init__(len(self.image_l))

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        frame_ns_cam = self.frame_timestamps[index]
        prev_ns_cam = self.frame_timestamps[index - 1] if index > 0 else (frame_ns_cam - self.imu_window_ns)

        frame_ns_imu = frame_ns_cam - self.imu_time_offset_ns
        prev_ns_imu = prev_ns_cam - self.imu_time_offset_ns

        imu_time_ns, imu_acc, imu_gyro, imu_sampling_map = (
            self.imu_loader.query_range_with_sampling_map(prev_ns_imu, frame_ns_imu)
        )
        if imu_time_ns.numel() == 0:
            _, nearest_time_ns, nearest_acc, nearest_gyro = self.imu_loader.query_nearest(frame_ns_imu)
            aligned_nearest_time_ns = int(nearest_time_ns[0].item()) + self.imu_time_offset_ns
            dt_ns = abs(aligned_nearest_time_ns - frame_ns_cam)

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
            time_ns=[frame_ns_cam],
            stereo=StereoData(
                T_BS=self.cam_T_BS,
                K=self.K,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                width=image_l.size(-1),
                height=image_l.size(-2),
                time_ns=[frame_ns_cam],
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
            frame.imu_calib_sigma_unit = self.imu_calib_sigma_unit
            frame.imu_calib_source = self.imu_calib_source
            frame.imu_calib_measurement_rate_hz = self.imu_calib_measurement_rate_hz
        if imu_sampling_map is not None:
            frame.imu_sampling_raw_time_ns = imu_sampling_map.raw_time_ns
            frame.imu_sampling_knot_from_raw = imu_sampling_map.knot_from_raw
        if self.imu_calib_acc_w_sigma is not None:
            frame.imu_calib_acc_w_sigma = self.imu_calib_acc_w_sigma
            frame.imu_calib_gyro_w_sigma = self.imu_calib_gyro_w_sigma
            frame.imu_calib_bias_random_walk_update_hz = self.imu_calib_bias_random_walk_update_hz
        frame.imu_source_world_frame = self.imu_source_world_frame
        frame.imu_source_measurement_frame = self.imu_source_measurement_frame
        frame.imu_internal_world_frame = self.imu_internal_world_frame
        frame.imu_internal_measurement_frame = self.imu_internal_measurement_frame
        frame.imu_world_frame = self.imu_world_frame
        frame.imu_measurement_frame = self.imu_measurement_frame
        frame.imu_acc_unit = self.imu_acc_unit
        frame.imu_gyro_unit = self.imu_gyro_unit
        frame.imu_timestamp_unit = self.imu_timestamp_unit
        frame.imu_time_offset_ns = self.imu_time_offset_ns
        frame.imu_time_offset_source = self.imu_time_offset_source
        frame.imu_metadata_gravity_m_s2 = self.imu_metadata_gravity_m_s2
        frame.imu_gravity_source = self.gravity_source
        frame.imu_acc_includes_gravity = self.imu_acc_includes_gravity
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
            "auto_estimate_time_offset": lambda v: isinstance(v, bool),
            "imu_time_offset_ns": lambda v: isinstance(v, int),
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
        sigma_unit = imu_calib.get("sigma_unit", "legacy_sqrt_rate_scaled")
        lines.append(f"  imu noise: σ_a={imu_calib.get('AccelSigma')}, "
                      f"σ_g={imu_calib.get('AngVelSigma')} "
                      f"(sigma_unit={sigma_unit}, from {src})")
    else:
        lines.append("  imu noise: from config defaults (no calibration)")

    if imu_extrinsic:
        t_ic = imu_extrinsic.get("T_imu_camera", {})
        trans_ic = t_ic.get("translation_body_nwu_m", "N/A")
        rot_ic = t_ic.get("rotation", "N/A")
        lines.append(f"  camera extrinsic: T_imu_camera translation={trans_ic}, rotation={rot_ic}")
        if meta and _metadata_uses_holoocean_nwu_flu(meta):
            t_bi = imu_extrinsic.get("T_body_imu", {})
            trans_bi = t_bi.get("translation_body_nwu_m", None)
            t_bc = imu_extrinsic.get("T_body_camera", {})
            trans_bc = t_bc.get("translation_body_nwu_m", None)
            if trans_bi and len(trans_bi) == 3:
                trans_internal = _body_nwu_translation_to_internal_ned(trans_bi)
                lines.append(
                    "  imu_T_BS: T_body_imu "
                    f"body_NWU={trans_bi} -> internal NED={trans_internal}, "
                    "rotation=R_x(180°)"
                )
                if trans_bc and len(trans_bc) == 3:
                    camera_imu_nwu = [float(trans_bi[i]) - float(trans_bc[i]) for i in range(3)]
                    camera_imu_internal = _body_nwu_translation_to_internal_ned(camera_imu_nwu)
                    lines.append(
                        "  imu_vio_sensor_T_imu: T_body_imu - T_body_camera "
                        f"camera_NWU={camera_imu_nwu} -> internal NED={camera_imu_internal}"
                    )
            else:
                lines.append("  imu_T_BS: T_body_imu missing, using T_imu_camera fallback")
    else:
        lines.append("  extrinsic: identity (no metadata)")

    if meta:
        cc = meta.get("coordinate_convention", {})
        imu_meta = meta.get("imu", {})
        gt_meta = meta.get("ground_truth", {})
        ts = meta.get("time_synchronization", {})
        lines.append(f"  coord: world={cc.get('export_world_frame','?')}, "
                     f"body={cc.get('body_frame','?')}, imu={cc.get('imu_measurement_frame', cc.get('imu_frame','?'))}")
        lines.append(f"  imu meas: frame={imu_meta.get('frame','?')}, "
                     f"acc_unit={imu_meta.get('acc_unit','?')}, "
                     f"gyro_unit={imu_meta.get('gyro_unit','?')}, "
                     f"acc_incl_g={imu_meta.get('acc_includes_gravity','?')}, "
                     f"gravity={imu_meta.get('gravity_m_s2','?')}")
        if _metadata_uses_holoocean_nwu_flu(meta):
            lines.append("  IMU frame: source FLU/body-NWU converted to MACVO internal NED via R_x(180°)")
        else:
            lines.append("  FLU→NED: applied via imu_T_BS rotation R_x(180°)")
        lines.append(f"  GT quat: meaning={gt_meta.get('quaternion_meaning','?')}, "
                     f"order={gt_meta.get('quaternion_order','?')}")
        lines.append(f"  GT pos frame: {gt_meta.get('position_frame','?')}")
        lines.append(
            "  GT velocity frame: "
            f"{gt_meta.get('velocity_frame', cc.get('ref_pose_velocity_frame','?'))}"
        )
        lines.append(f"  timestamp unit: {ts.get('timestamp_unit','?')}, "
                     f"camera_imu_offset_ns={ts.get('camera_imu_time_offset_ns','?')}")

    for line in lines:
        Logger.write("info", line)
