import torch
import traceback
import numpy as np
import pypose as pp
import json

from typing import Callable, Generic
from typing_extensions import Self
from abc import ABC, abstractmethod

from Utility.Sandbox import Sandbox
from DataLoader import SequenceBase, T_Data
from Module.Map import VisualMap
from Utility.PoseFrame import (
    convert_pose_frame,
    convert_pose_world_frame_only,
    write_timed_se3_csv,
)
from Utility.TrajectoryReference import (
    compose_camera_to_imu_poses,
    constant_camera_T_imu,
)
from Utility.PrettyPrint import ColoredTqdm, Logger

from torch.profiler import profile, ProfilerActivity


class IOdometry(ABC, Generic[T_Data]):
    def __init__(self, profile: bool = False) -> None:
        super().__init__()
        self.terminated = False
        self.profile    = profile
        self.profile_save_path = "trace_parallel.json"
        self._profiled_once = False
    
    def receive_frames(self, sequence: SequenceBase[T_Data], saveto: Sandbox, on_frame_finished: None | Callable[[T_Data, Self, ColoredTqdm], None]=None):
        try:
            reference_poses, reference_time = self._run_sequence(sequence, on_frame_finished)
            global_map = self.get_map()
            
            pose_output_frame = str(getattr(sequence, "pose_output_frame", "NED")).upper()

            sensor_poses = pp.SE3(global_map.frames.data["pose"].tensor)
            T_BS         = pp.SE3(global_map.frames.data["T_BS"].tensor)
            body_poses: np.ndarray = (T_BS @ sensor_poses @ T_BS.Inv()).tensor().cpu().numpy()
            body_poses = convert_pose_frame(body_poses, "NED", pose_output_frame)
            time_ns   : np.ndarray = global_map.frames.data["time_ns"].tensor.cpu().numpy()

            write_timed_se3_csv(saveto.path("poses.csv"), time_ns, body_poses)
            saveto.path("pose_coordinate_frame.txt").write_text(pose_output_frame + "\n", encoding="utf-8")

            # Keep Map pose semantics at the visual sensor origin, but also
            # export the exact IMU-origin trajectory used by the VIO factors.
            # T_CI is metadata-derived and stored on every frame; composing it
            # here avoids changing the pose that MACVO expects on the next edge.
            if "imu_vio_sensor_T_imu" in global_map.frames.data:
                camera_T_imu_all = (
                    global_map.frames.data["imu_vio_sensor_T_imu"].tensor
                    .detach().cpu().double().numpy()
                )
                camera_poses_internal = sensor_poses.tensor().detach().cpu().double().numpy()
                if camera_T_imu_all.shape[0] != camera_poses_internal.shape[0]:
                    raise ValueError(
                        "camera-to-IMU extrinsic count does not match trajectory: "
                        f"{camera_T_imu_all.shape[0]} vs {camera_poses_internal.shape[0]}"
                    )
                camera_T_imu = constant_camera_T_imu(camera_T_imu_all)
                imu_poses_internal = compose_camera_to_imu_poses(
                    camera_poses_internal,
                    camera_T_imu_all,
                )
                imu_poses_output = convert_pose_world_frame_only(
                    imu_poses_internal,
                    "NED",
                    pose_output_frame,
                )
                write_timed_se3_csv(
                    saveto.path("poses_imu.csv"),
                    time_ns,
                    imu_poses_output,
                )
                saveto.path("pose_reference_points.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "poses.csv": "visual_sensor_origin (camera for current HoloOcean runs)",
                            "poses_imu.csv": "IMU origin used by VIO state T_WI",
                            "runtime_extrinsic_field": "frames//imu_vio_sensor_T_imu",
                            "runtime_extrinsic_semantics": (
                                "T_CI maps raw IMU frame I to MACVO camera frame C; "
                                "p_C = T_CI p_I and T_WI = T_WC * T_CI"
                            ),
                            "runtime_imu_local_frame": "raw IMU FLU",
                            "runtime_world_frame": "NED",
                            "runtime_T_CI_xyzw": camera_T_imu.tolist(),
                            "output_world_frame": pose_output_frame,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            np.savez_compressed(saveto.path("tensor_map.npz"), **global_map.serialize())
            if hasattr(self, "export_diagnostics"):
                self.export_diagnostics(saveto)  # type: ignore[attr-defined]
            
            if len(reference_poses) > 1:    # At least two poses for a non-trivial trajectory
                ref_body_poses: np.ndarray = torch.cat(reference_poses, dim=0).numpy()
                ref_body_poses = convert_pose_frame(ref_body_poses, "NED", pose_output_frame)
                ref_time_ns   : np.ndarray = np.array(reference_time, dtype=np.int64)
                write_timed_se3_csv(saveto.path("ref_poses.csv"), ref_time_ns, ref_body_poses)
            else:
                Logger.write("warn", f"Did not write {saveto.path('ref_poses.csv')} since less than 2 ground truth poses in sequence.")
            
        except KeyboardInterrupt as e:
            Logger.write("fatal", f"Experiment at {saveto.folder} is interrupted.")
            raise e
        except Exception:
            Logger.show_exception()
            Logger.write("fatal", f"Failed to execute experiment at {saveto.folder}.")

    def _run_sequence(
        self,
        sequence: SequenceBase[T_Data],
        on_frame_finished: None | Callable[[T_Data, Self, ColoredTqdm], None],
    ) -> tuple[list, list]:
        reference_poses, reference_time = [], []
        primary_error: BaseException | None = None
        try:
            pb = ColoredTqdm(sequence)
            frame: T_Data
            for frame in pb:
                if self.profile and (not self._profiled_once):
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, with_flops=True) as prof:
                        self.run(frame)
                    prof.export_chrome_trace(self.profile_save_path)
                    self._profiled_once = True
                else:
                    self.run(frame)

                if frame.gt_pose is not None:
                    reference_poses.append(frame.gt_pose)
                    reference_time.append(frame.time_ns[0])

                if on_frame_finished is not None:
                    on_frame_finished(frame, self, pb)

            self.validate_completion()
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self.terminate()
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                Logger.write(
                    "error",
                    f"Cleanup failed while preserving {type(primary_error).__name__}: {cleanup_error}",
                )
        return reference_poses, reference_time
        
    @abstractmethod
    def run(self, frame: T_Data) -> None:
        """
        Core method for IVisualOdometry. This method handles the incoming frames and perform tracking/mapping internally.
        """
        ...
    
    @abstractmethod
    def get_map(self) -> VisualMap:
        """
        Provides the VisualMap built across multiple calls of .run(...).
        """
        ...

    def validate_completion(self) -> None:
        """Validate a normally exhausted input sequence before final cleanup."""
        return None

    def terminate(self) -> None: 
        """
        You can define additional operations on terminate. For instance, smoothing trajectory / interpolate bad frames etc.
        """
        self.terminated = True
