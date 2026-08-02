#include "T2ISAM2Backend.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
using t2_isam2::T2BackendEdge;
using t2_isam2::T2BackendState;
using t2_isam2::T2BackendUpdate;
using t2_isam2::T2ISAM2Backend;

namespace {

gtsam::Pose3 PoseFromXYZW(const gtsam::Vector& value) {
  if (value.size() != 7 || !value.allFinite()) {
    throw std::invalid_argument("pose must contain finite [t, q_xyzw] values");
  }
  return gtsam::Pose3(
      gtsam::Rot3::Quaternion(value(6), value(3), value(4), value(5)),
      gtsam::Point3(value(0), value(1), value(2)));
}

gtsam::Vector PoseToXYZW(const gtsam::Pose3& pose) {
  gtsam::Vector result(7);
  const auto quaternion = pose.rotation().toQuaternion();
  result << pose.x(), pose.y(), pose.z(), quaternion.x(), quaternion.y(),
      quaternion.z(), quaternion.w();
  return result;
}

gtsam::imuBias::ConstantBias BiasFromVectors(
    const gtsam::Vector& acc, const gtsam::Vector& gyro) {
  if (acc.size() != 3 || gyro.size() != 3) {
    throw std::invalid_argument("bias vectors must contain three values");
  }
  return gtsam::imuBias::ConstantBias(acc, gyro);
}

T2BackendState StateFromDict(const py::dict& payload, const std::string& prefix) {
  T2BackendState state;
  state.pose_WB = PoseFromXYZW(
      payload[(prefix + "_pose_WB").c_str()].cast<gtsam::Vector>());
  state.velocity_W =
      payload[(prefix + "_velocity_W").c_str()].cast<gtsam::Vector3>();
  state.bias = BiasFromVectors(
      payload[(prefix + "_acc_bias").c_str()].cast<gtsam::Vector>(),
      payload[(prefix + "_gyro_bias").c_str()].cast<gtsam::Vector>());
  return state;
}

py::dict StateToDict(const T2BackendState& state) {
  py::dict result;
  result["local_index"] = state.local_index;
  result["frame_idx"] = state.frame;
  result["pose_WB"] = PoseToXYZW(state.pose_WB);
  result["velocity_W"] = state.velocity_W;
  result["acc_bias"] = state.bias.accelerometer();
  result["gyro_bias"] = state.bias.gyroscope();
  return result;
}

T2BackendEdge EdgeFromDict(const py::dict& payload) {
  T2BackendEdge edge;
  edge.frame_i = payload["frame_i"].cast<int>();
  edge.frame_j = payload["frame_j"].cast<int>();
  edge.initial_i = StateFromDict(payload, "state_i");
  edge.initial_j = StateFromDict(payload, "state_j");
  edge.initial_i.frame = edge.frame_i;
  edge.initial_j.frame = edge.frame_j;
  edge.delta_rotation = gtsam::Rot3::Expmap(
      payload["imu_delta_rotvec"].cast<gtsam::Vector3>());
  edge.delta_velocity =
      payload["imu_delta_velocity"].cast<gtsam::Vector3>();
  edge.delta_position =
      payload["imu_delta_position"].cast<gtsam::Vector3>();
  edge.imu_covariance =
      payload["imu_covariance_pvr"].cast<gtsam::Matrix9>();
  edge.dt = payload["imu_dt"].cast<double>();
  edge.bias_jacobian =
      payload["imu_bias_jacobian_pvr_babg"]
          .cast<Eigen::Matrix<double, 9, 6>>();
  edge.linearized_acc_bias =
      payload["imu_linearized_acc_bias"].cast<gtsam::Vector3>();
  edge.linearized_gyro_bias =
      payload["imu_linearized_gyro_bias"].cast<gtsam::Vector3>();
  edge.bias_rw_covariance =
      payload["bias_rw_covariance_babg"]
          .cast<Eigen::Matrix<double, 6, 6>>();
  const py::handle gravity = payload["gravity_world"];
  if (!gravity.is_none()) {
    edge.gravity_world = gravity.cast<gtsam::Vector3>();
  }
  edge.gravity_in_residual =
      payload["gravity_handling"].cast<std::string>() == "residual";
  edge.visual_factor_mode =
      payload["visual_factor_mode"].cast<std::string>();
  if (edge.visual_factor_mode == "compressed_uvd") {
    edge.visual_reference_CjCi = PoseFromXYZW(
        payload["visual_reference_CjCi"].cast<gtsam::Vector>());
    edge.visual_A =
        payload["visual_sqrt_information"].cast<gtsam::Matrix>();
    edge.visual_c =
        payload["visual_residual_offset"].cast<gtsam::Vector>();
  } else if (edge.visual_factor_mode == "relative_pose") {
    edge.visual_measurement_BiBj = PoseFromXYZW(
        payload["visual_measurement_BiBj"].cast<gtsam::Vector>());
    edge.visual_covariance_tr =
        payload["visual_covariance_tr"].cast<gtsam::Matrix6>();
    edge.visual_huber_delta = payload["visual_huber_delta"].cast<double>();
  } else if (edge.visual_factor_mode == "direct_uvd") {
    edge.visual_points_Ci =
        payload["visual_points_Ci"].cast<gtsam::Matrix>();
    edge.visual_target_uvd =
        payload["visual_target_uvd"].cast<gtsam::Matrix>();
    edge.visual_covariance_uvd_flat =
        payload["visual_covariance_uvd_flat"].cast<gtsam::Matrix>();
    edge.visual_intrinsic =
        payload["visual_intrinsic"].cast<Eigen::Matrix3d>();
    edge.visual_baseline = payload["visual_baseline"].cast<double>();
    edge.visual_huber_delta = payload["visual_huber_delta"].cast<double>();
  } else {
    throw std::invalid_argument(
        "unsupported iSAM2 visual factor mode: " + edge.visual_factor_mode);
  }
  edge.extrinsic_CI =
      PoseFromXYZW(payload["extrinsic_CI"].cast<gtsam::Vector>());
  const bool has_velocity_prior_mean =
      payload.contains("velocity_prior_mean_W");
  const bool has_velocity_prior_covariance =
      payload.contains("velocity_prior_covariance_W");
  if (has_velocity_prior_mean != has_velocity_prior_covariance) {
    throw std::invalid_argument(
        "velocity prior requires both mean and covariance");
  }
  if (has_velocity_prior_mean) {
    edge.velocity_prior_enabled = true;
    edge.velocity_prior_mean_W =
        payload["velocity_prior_mean_W"].cast<gtsam::Vector3>();
    edge.velocity_prior_covariance =
        payload["velocity_prior_covariance_W"].cast<Eigen::Matrix3d>();
  }
  return edge;
}

}  // namespace

PYBIND11_MODULE(pace_vio_isam2_backend, module) {
  module.doc() = "Incremental iSAM2 backend for PACE-VIO factor packets";
  py::class_<T2ISAM2Backend>(module, "PACEISAM2Backend")
      .def(
          py::init<double, int, double>(),
          py::arg("relinearize_threshold") = 0.01,
          py::arg("relinearize_skip") = 1,
          py::arg("covariance_floor") = 1.0e-12)
      .def(
          "reset",
          [](T2ISAM2Backend& backend, const py::dict& payload,
             const gtsam::Vector& prior_sigma_t2) {
            T2BackendState state = StateFromDict(payload, "state_i");
            state.frame = payload["frame_i"].cast<int>();
            backend.Reset(
                state.frame, state, prior_sigma_t2,
                PoseFromXYZW(payload["extrinsic_CI"].cast<gtsam::Vector>()));
            return StateToDict(backend.LatestState());
          },
          py::arg("factor_packet"), py::arg("prior_sigma_t2"))
      .def(
          "add_edge",
          [](T2ISAM2Backend& backend, const py::dict& payload) {
            const T2BackendUpdate update = backend.AddEdge(EdgeFromDict(payload));
            py::dict result = StateToDict(update.latest);
            result["previous_state"] = StateToDict(update.previous);
            result["update_ms"] = update.update_ms;
            result["imu_cost"] = update.imu_cost;
            result["bias_cost"] = update.bias_cost;
            result["visual_cost"] = update.visual_cost;
            result["velocity_prior_cost"] = update.velocity_prior_cost;
            result["total_edge_cost"] =
                update.imu_cost + update.bias_cost + update.visual_cost +
                update.velocity_prior_cost;
            result["initial_pose_mismatch_norm"] =
                update.initial_pose_mismatch_norm;
            result["initial_velocity_mismatch_norm"] =
                update.initial_velocity_mismatch_norm;
            result["initial_bias_mismatch_norm"] =
                update.initial_bias_mismatch_norm;
            return result;
          },
          py::arg("factor_packet"))
      .def("latest_state", [](const T2ISAM2Backend& backend) {
        return StateToDict(backend.LatestState());
      })
      .def("history", [](const T2ISAM2Backend& backend) {
        py::list result;
        for (const auto& state : backend.History()) {
          result.append(StateToDict(state));
        }
        return result;
      })
      .def_property_readonly("initialized", &T2ISAM2Backend::initialized)
      .def_property_readonly("latest_frame", &T2ISAM2Backend::latestFrame)
      .def_property_readonly("state_count", &T2ISAM2Backend::stateCount);
  module.attr("T2ISAM2Backend") = module.attr("PACEISAM2Backend");
}
