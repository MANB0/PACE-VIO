from __future__ import annotations

import numpy as np
import pypose as pp
import torch

from Utility.PoseFrame import convert_pose_frame, convert_pose_world_frame_only


def compose_camera_to_imu_poses(
    camera_poses: np.ndarray,
    camera_T_imu: np.ndarray,
) -> np.ndarray:
    """Compose ``T_WI = T_WC * T_CI`` for one transform per pose or one constant.

    ``T_WC`` uses MACVO's internal world/camera NED axes. ``T_CI`` maps the
    raw IMU frame I into that camera frame C, so their adjacent C frames are
    compatible even when I itself is FLU.
    """
    poses = np.asarray(camera_poses, dtype=np.float64)
    extrinsic = np.asarray(camera_T_imu, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"camera poses must have shape Nx7, got {poses.shape}")
    if extrinsic.shape == (7,):
        extrinsic = np.broadcast_to(extrinsic.reshape(1, 7), poses.shape).copy()
    if extrinsic.shape != poses.shape:
        raise ValueError(
            "camera-to-IMU extrinsic must have shape 7 or Nx7 matching poses; "
            f"got {extrinsic.shape} for {poses.shape}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(extrinsic).all():
        raise ValueError("camera poses and camera-to-IMU extrinsics must be finite")
    pose_tensor = torch.from_numpy(poses)
    extrinsic_tensor = torch.from_numpy(extrinsic)
    return (pp.SE3(pose_tensor) @ pp.SE3(extrinsic_tensor)).tensor().cpu().numpy()


def translate_pose_reference_point(
    poses: np.ndarray,
    reference_to_target_translation: np.ndarray,
) -> np.ndarray:
    """Move pose positions from one rigid-body point to another.

    ``poses`` contain ``T_WR`` with an ``R``-aligned orientation and
    ``reference_to_target_translation`` is ``r_RT`` expressed in that local
    orientation. The returned positions satisfy ``p_WT = p_WR + R_WR r_RT``;
    orientation is unchanged. This is the appropriate operation when only the
    physical reference point changes (camera, body origin, or IMU origin).
    """
    values = np.asarray(poses, dtype=np.float64)
    translation = np.asarray(reference_to_target_translation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"poses must have shape Nx7, got {values.shape}")
    if translation.shape == (3,):
        translation = np.broadcast_to(
            translation.reshape(1, 3), (values.shape[0], 3)
        ).copy()
    if translation.shape != (values.shape[0], 3):
        raise ValueError(
            "reference-to-target translation must have shape 3 or Nx3; "
            f"got {translation.shape} for {values.shape}"
        )
    if not np.isfinite(values).all() or not np.isfinite(translation).all():
        raise ValueError("poses and reference-point translations must be finite")

    rotation = pp.SO3(torch.from_numpy(values[:, 3:7]))
    rotated_translation = rotation.Act(torch.from_numpy(translation)).cpu().numpy()
    result = values.copy()
    result[:, :3] += rotated_translation
    return result


def rebase_pose_positions(poses: np.ndarray, *, index: int = 0) -> np.ndarray:
    """Translate a trajectory so its selected physical point starts at zero.

    This intentionally preserves the NWU world axes and every orientation. It
    removes only the arbitrary global translation introduced when MACVO sets
    its first camera pose to identity.
    """
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] == 0:
        raise ValueError(f"poses must have non-empty shape Nx7, got {values.shape}")
    if not 0 <= index < values.shape[0]:
        raise IndexError(f"rebase index {index} is outside {values.shape[0]} poses")
    result = values.copy()
    result[:, :3] -= result[index, :3]
    return result


def rigid_body_velocity_at_target(
    reference_velocity_world: np.ndarray,
    angular_velocity_local: np.ndarray,
    reference_orientation_xyzw: np.ndarray,
    reference_to_target_translation: np.ndarray,
) -> np.ndarray:
    """Move rigid-body velocity from reference point R to target point T."""
    velocity = np.asarray(reference_velocity_world, dtype=np.float64)
    omega = np.asarray(angular_velocity_local, dtype=np.float64)
    quaternion = np.asarray(reference_orientation_xyzw, dtype=np.float64)
    translation = np.asarray(reference_to_target_translation, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"reference velocity must have shape Nx3, got {velocity.shape}")
    if omega.shape != velocity.shape or quaternion.shape != (velocity.shape[0], 4):
        raise ValueError("velocity, angular velocity, and quaternion row counts must match")
    if translation.shape == (3,):
        translation = np.broadcast_to(
            translation.reshape(1, 3), velocity.shape
        ).copy()
    if translation.shape != velocity.shape:
        raise ValueError("reference-to-target translation must contain one row per velocity")
    rotation_world_reference = pp.SO3(torch.from_numpy(quaternion)).matrix().cpu().numpy()
    lever_velocity_local = np.cross(omega, translation)
    lever_velocity_world = np.einsum(
        "nij,nj->ni", rotation_world_reference, lever_velocity_local
    )
    return velocity + lever_velocity_world


def camera_nwu_poses_to_imu_nwu(
    camera_poses_nwu: np.ndarray,
    camera_T_imu_internal_ned: np.ndarray,
) -> np.ndarray:
    """Move NWU camera-origin poses to the IMU origin using the runtime T_CI."""
    camera_ned = convert_pose_frame(camera_poses_nwu, "NWU", "NED")
    imu_ned = compose_camera_to_imu_poses(camera_ned, camera_T_imu_internal_ned)
    return convert_pose_world_frame_only(imu_ned, "NED", "NWU")


def camera_velocity_to_imu_velocity_nwu(
    camera_velocity_world_nwu: np.ndarray,
    angular_velocity_body_nwu: np.ndarray,
    camera_orientation_xyzw_nwu: np.ndarray,
    camera_T_imu_internal_ned: np.ndarray,
) -> np.ndarray:
    """Apply the rigid-body lever-arm velocity correction at the IMU origin.

    ``v_I = v_C + R_WC (omega_C x r_CI)``. The runtime extrinsic translation
    is converted from internal NED coordinates back to camera/body NWU before
    applying the correction.
    """
    velocity = np.asarray(camera_velocity_world_nwu, dtype=np.float64)
    omega = np.asarray(angular_velocity_body_nwu, dtype=np.float64)
    quaternion = np.asarray(camera_orientation_xyzw_nwu, dtype=np.float64)
    extrinsic = np.asarray(camera_T_imu_internal_ned, dtype=np.float64).reshape(-1, 7)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"camera velocity must have shape Nx3, got {velocity.shape}")
    if omega.shape != velocity.shape or quaternion.shape != (velocity.shape[0], 4):
        raise ValueError("velocity, angular velocity, and quaternion row counts must match")
    if extrinsic.shape[0] not in (1, velocity.shape[0]):
        raise ValueError("extrinsic must contain one row or one row per velocity")
    if extrinsic.shape[0] == 1:
        extrinsic = np.broadcast_to(extrinsic, (velocity.shape[0], 7)).copy()

    # A vector expressed in internal NED is represented in body NWU by the
    # same x component and negated y/z components.
    lever_nwu = extrinsic[:, :3].copy()
    lever_nwu[:, 1:3] *= -1.0
    return rigid_body_velocity_at_target(
        velocity,
        omega,
        quaternion,
        lever_nwu,
    )


def constant_camera_T_imu(extrinsics: np.ndarray, *, atol: float = 1e-9) -> np.ndarray:
    """Return one runtime T_CI after verifying it is finite and frame-constant."""
    values = np.asarray(extrinsics, dtype=np.float64).reshape(-1, 7)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("runtime camera-to-IMU extrinsic is empty or non-finite")
    reference = values[0]
    if not np.allclose(values, reference.reshape(1, 7), atol=atol, rtol=0.0):
        raise ValueError("runtime camera-to-IMU extrinsic changes across frames")
    return reference.copy()
