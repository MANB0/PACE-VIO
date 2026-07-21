import torch
import pypose as pp
from pathlib import Path

from Utility.IMUKinematics import compose_adaptive_fallback_pose


def _se3(translation, rotvec):
    trans = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
    quat = pp.so3(torch.tensor(rotvec, dtype=torch.float32).reshape(1, 3)).Exp().tensor()
    return pp.SE3(torch.cat([trans, quat], dim=-1))


def _relative(prev_pose, pose):
    return pp.SE3(prev_pose).Inv() @ pp.SE3(pose)


def test_fallback_uses_imu_rotation_and_translation_when_both_enabled():
    prev_pose = _se3([1.0, 2.0, 3.0], [0.0, 0.0, 0.1])
    visual_pose = prev_pose @ _se3([10.0, 0.0, 0.0], [0.0, 0.0, 0.2])
    imu_rel_pose = _se3([0.5, -0.1, 0.2], [0.0, 0.0, 0.4])

    fallback = compose_adaptive_fallback_pose(
        prev_pose=prev_pose,
        visual_pose=visual_pose,
        imu_rel_pose=imu_rel_pose,
        use_imu_rotation=True,
        use_imu_translation=True,
    )

    rel = _relative(prev_pose, fallback)
    assert torch.allclose(rel.translation().reshape(3), torch.tensor([0.5, -0.1, 0.2]), atol=1e-5)
    assert torch.allclose(rel.rotation().Log().tensor().reshape(3), torch.tensor([0.0, 0.0, 0.4]), atol=1e-5)


def test_fallback_keeps_visual_translation_when_translation_is_disabled():
    prev_pose = _se3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    visual_pose = prev_pose @ _se3([2.0, 0.0, 0.0], [0.0, 0.0, 0.2])
    imu_rel_pose = _se3([0.1, 0.2, 0.3], [0.0, 0.0, 0.5])

    fallback = compose_adaptive_fallback_pose(
        prev_pose=prev_pose,
        visual_pose=visual_pose,
        imu_rel_pose=imu_rel_pose,
        use_imu_rotation=True,
        use_imu_translation=False,
    )

    rel = _relative(prev_pose, fallback)
    assert torch.allclose(rel.translation().reshape(3), torch.tensor([2.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(rel.rotation().Log().tensor().reshape(3), torch.tensor([0.0, 0.0, 0.5]), atol=1e-5)


def test_fallback_returns_visual_pose_when_imu_is_unavailable():
    visual_pose = _se3([2.0, 1.0, -0.5], [0.1, 0.0, 0.0])

    fallback = compose_adaptive_fallback_pose(
        prev_pose=_se3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        visual_pose=visual_pose,
        imu_rel_pose=None,
        use_imu_rotation=True,
        use_imu_translation=True,
    )

    assert torch.allclose(fallback.tensor(), visual_pose.tensor(), atol=1e-6)


def test_fallback_converts_imu_relative_pose_to_camera_relative_pose_with_lever_arm():
    prev_pose = _se3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    visual_pose = _se3([5.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    sensor_T_imu = _se3([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    yaw = 1.5707963267948966
    imu_rel_pose = _se3([-1.0, 1.0, 0.0], [0.0, 0.0, yaw])

    fallback = compose_adaptive_fallback_pose(
        prev_pose=prev_pose,
        visual_pose=visual_pose,
        imu_rel_pose=imu_rel_pose,
        use_imu_rotation=True,
        use_imu_translation=True,
        sensor_T_imu=sensor_T_imu,
    )

    rel = _relative(prev_pose, fallback)
    assert torch.allclose(rel.translation().reshape(3), torch.zeros(3), atol=1e-5)
    assert torch.allclose(rel.rotation().Log().tensor().reshape(3), torch.tensor([0.0, 0.0, yaw]), atol=1e-5)


def test_lost_track_fallback_call_uses_pose_before_prev_keyframe_update():
    source = Path("Odometry/MACVO.py").read_text(encoding="utf-8")

    assert 'prev_pose_for_fallback = self.graph.frames.data["pose"][self.prev_keyframe[1]]' not in source
