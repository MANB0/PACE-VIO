#pragma once

#include "T2Factors.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace t2_isam2 {

using gtsam::Matrix;
using gtsam::Matrix6;
using gtsam::Matrix9;
using gtsam::NonlinearFactorGraph;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Values;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::Vector6;
using gtsam::imuBias::ConstantBias;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

struct T2BackendState {
  int local_index{};
  int frame{};
  Pose3 pose_WB;
  Vector3 velocity_W = Vector3::Zero();
  ConstantBias bias;
};

struct T2BackendEdge {
  int frame_i{};
  int frame_j{};
  T2BackendState initial_i;
  T2BackendState initial_j;
  Rot3 delta_rotation;
  Vector3 delta_velocity = Vector3::Zero();
  Vector3 delta_position = Vector3::Zero();
  Matrix9 imu_covariance = Matrix9::Identity();
  double dt{};
  Eigen::Matrix<double, 9, 6> bias_jacobian =
      Eigen::Matrix<double, 9, 6>::Zero();
  Vector3 linearized_acc_bias = Vector3::Zero();
  Vector3 linearized_gyro_bias = Vector3::Zero();
  Eigen::Matrix<double, 6, 6> bias_rw_covariance =
      Eigen::Matrix<double, 6, 6>::Identity();
  Vector3 gravity_world = Vector3::Zero();
  bool gravity_in_residual{};
  Pose3 visual_reference_CjCi;
  Matrix visual_A;
  Vector visual_c;
  Pose3 extrinsic_CI;
};

struct T2BackendUpdate {
  T2BackendState previous;
  T2BackendState latest;
  double update_ms{};
  double imu_cost{};
  double bias_cost{};
  double visual_cost{};
  double initial_pose_mismatch_norm{};
  double initial_velocity_mismatch_norm{};
  double initial_bias_mismatch_norm{};
};

template <int N>
Eigen::Matrix<double, N, N> StabilizeCovariance(
    const Eigen::Matrix<double, N, N>& covariance, double floor) {
  const auto symmetric = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("covariance eigendecomposition failed");
  }
  auto eigenvalues = solver.eigenvalues();
  const double scale = std::max(eigenvalues.cwiseAbs().maxCoeff(), 1.0);
  const double threshold = std::max(
      floor, std::numeric_limits<double>::epsilon() * scale);
  for (int index = 0; index < N; ++index) {
    eigenvalues(index) = std::max(eigenvalues(index), threshold);
  }
  return solver.eigenvectors() * eigenvalues.asDiagonal() *
         solver.eigenvectors().transpose();
}

class T2ISAM2Backend {
 public:
  T2ISAM2Backend(
      double relinearize_threshold = 0.01, int relinearize_skip = 1,
      double covariance_floor = 1.0e-12)
      : relinearize_threshold_(relinearize_threshold),
        relinearize_skip_(relinearize_skip),
        covariance_floor_(covariance_floor) {
    if (!(relinearize_threshold_ > 0.0) || relinearize_skip_ < 1 ||
        !(covariance_floor_ > 0.0)) {
      throw std::invalid_argument("invalid T2 iSAM2 backend configuration");
    }
  }

  void Reset(
      int frame, const T2BackendState& initial_state,
      const Vector& prior_sigma_t2, const Pose3& extrinsic_CI) {
    if (prior_sigma_t2.size() != 15 ||
        !(prior_sigma_t2.array() > 0.0).all() ||
        !prior_sigma_t2.allFinite()) {
      throw std::invalid_argument("T2 prior sigma must contain 15 positive values");
    }
    gtsam::ISAM2Params params;
    params.relinearizeThreshold = relinearize_threshold_;
    params.relinearizeSkip = relinearize_skip_;
    params.factorization = gtsam::ISAM2Params::CHOLESKY;
    isam_ = std::make_unique<gtsam::ISAM2>(params);

    const Matrix6 permutation = TranslationRotationFromRotationTranslation();
    const Vector6 pose_sigma_rt = permutation * prior_sigma_t2.head<6>();
    NonlinearFactorGraph factors;
    factors.push_back(std::make_shared<gtsam::PriorFactor<Pose3>>(
        X(0), initial_state.pose_WB,
        gtsam::noiseModel::Diagonal::Sigmas(pose_sigma_rt)));
    factors.push_back(std::make_shared<gtsam::PriorFactor<Vector3>>(
        V(0), initial_state.velocity_W,
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigma_t2.segment<3>(6))));
    factors.push_back(std::make_shared<gtsam::PriorFactor<ConstantBias>>(
        B(0), initial_state.bias,
        gtsam::noiseModel::Diagonal::Sigmas(prior_sigma_t2.segment<6>(9))));

    Values values;
    values.insert(X(0), initial_state.pose_WB);
    values.insert(V(0), initial_state.velocity_W);
    values.insert(B(0), initial_state.bias);
    isam_->update(factors, values);

    frames_.clear();
    frames_.push_back(frame);
    extrinsic_CI_ = extrinsic_CI;
    initialized_ = true;
  }

  bool initialized() const { return initialized_; }
  int latestFrame() const {
    RequireInitialized();
    return frames_.back();
  }
  int stateCount() const { return static_cast<int>(frames_.size()); }

  T2BackendUpdate AddEdge(const T2BackendEdge& edge) {
    RequireInitialized();
    if (edge.frame_i != frames_.back() || edge.frame_j <= edge.frame_i) {
      throw std::invalid_argument("T2 factor packet is not continuous with iSAM2 history");
    }
    if (!(edge.dt > 0.0) || !std::isfinite(edge.dt)) {
      throw std::invalid_argument("T2 factor packet has invalid IMU dt");
    }
    if (edge.visual_A.cols() != 6 || edge.visual_A.rows() < 1 ||
        edge.visual_A.rows() > 6 || edge.visual_c.size() != edge.visual_A.rows()) {
      throw std::invalid_argument("T2 factor packet has invalid compressed visual shape");
    }
    const double extrinsic_error = Pose3::Logmap(
        extrinsic_CI_.between(edge.extrinsic_CI)).lpNorm<Eigen::Infinity>();
    if (extrinsic_error > 1.0e-10) {
      throw std::invalid_argument("T2 factor packet changed T_CI during a run");
    }

    const int local_i = static_cast<int>(frames_.size()) - 1;
    const int local_j = local_i + 1;
    const Values before_estimate = isam_->calculateEstimate();
    const Pose3 previous_pose = before_estimate.at<Pose3>(X(local_i));
    const Vector3 previous_velocity = before_estimate.at<Vector3>(V(local_i));
    const ConstantBias previous_bias = before_estimate.at<ConstantBias>(B(local_i));

    const Matrix9 imu_covariance = StabilizeCovariance<9>(
        edge.imu_covariance, covariance_floor_);
    const Eigen::Matrix<double, 6, 6> bias_covariance =
        StabilizeCovariance<6>(edge.bias_rw_covariance, covariance_floor_);
    auto imu = std::make_shared<T2CachedImuFactor>(
        X(local_i), V(local_i), B(local_i), X(local_j), V(local_j),
        edge.delta_rotation, edge.delta_velocity, edge.delta_position,
        edge.dt, imu_covariance, edge.bias_jacobian,
        edge.linearized_acc_bias, edge.linearized_gyro_bias,
        edge.gravity_world, edge.gravity_in_residual);
    auto bias = std::make_shared<gtsam::BetweenFactor<ConstantBias>>(
        B(local_i), B(local_j), ConstantBias(),
        gtsam::noiseModel::Gaussian::Covariance(bias_covariance));
    auto visual = std::make_shared<T2CompressedVisualFactor>(
        X(local_i), X(local_j), edge.visual_reference_CjCi,
        edge.visual_A, edge.visual_c, edge.extrinsic_CI);

    NonlinearFactorGraph factors;
    factors.push_back(imu);
    factors.push_back(bias);
    factors.push_back(visual);
    Values values;
    values.insert(X(local_j), edge.initial_j.pose_WB);
    values.insert(V(local_j), edge.initial_j.velocity_W);
    values.insert(B(local_j), edge.initial_j.bias);

    const auto started = std::chrono::steady_clock::now();
    isam_->update(factors, values);
    const double update_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    frames_.push_back(edge.frame_j);

    const Values estimate = isam_->calculateEstimate();
    T2BackendUpdate result;
    result.previous = StateAt(estimate, local_i);
    result.latest = StateAt(estimate, local_j);
    result.update_ms = update_ms;
    result.imu_cost = imu->error(estimate);
    result.bias_cost = bias->error(estimate);
    result.visual_cost = visual->error(estimate);
    result.initial_pose_mismatch_norm = Pose3::Logmap(
        previous_pose.between(edge.initial_i.pose_WB)).norm();
    result.initial_velocity_mismatch_norm =
        (previous_velocity - edge.initial_i.velocity_W).norm();
    Vector6 previous_bias_vector;
    previous_bias_vector.head<3>() = previous_bias.accelerometer();
    previous_bias_vector.tail<3>() = previous_bias.gyroscope();
    Vector6 packet_bias_vector;
    packet_bias_vector.head<3>() = edge.initial_i.bias.accelerometer();
    packet_bias_vector.tail<3>() = edge.initial_i.bias.gyroscope();
    result.initial_bias_mismatch_norm =
        (previous_bias_vector - packet_bias_vector).norm();
    return result;
  }

  T2BackendState LatestState() const {
    RequireInitialized();
    const Values estimate = isam_->calculateEstimate();
    return StateAt(estimate, static_cast<int>(frames_.size()) - 1);
  }

  std::vector<T2BackendState> History() const {
    RequireInitialized();
    const Values estimate = isam_->calculateEstimate();
    std::vector<T2BackendState> states;
    states.reserve(frames_.size());
    for (int local = 0; local < static_cast<int>(frames_.size()); ++local) {
      states.push_back(StateAt(estimate, local));
    }
    return states;
  }

 private:
  void RequireInitialized() const {
    if (!initialized_ || !isam_) {
      throw std::runtime_error("T2 iSAM2 backend is not initialized");
    }
  }

  T2BackendState StateAt(const Values& estimate, int local) const {
    T2BackendState result;
    result.local_index = local;
    result.frame = frames_.at(local);
    result.pose_WB = estimate.at<Pose3>(X(local));
    result.velocity_W = estimate.at<Vector3>(V(local));
    result.bias = estimate.at<ConstantBias>(B(local));
    return result;
  }

  double relinearize_threshold_{};
  int relinearize_skip_{};
  double covariance_floor_{};
  bool initialized_{};
  Pose3 extrinsic_CI_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  std::vector<int> frames_;
};

}  // namespace t2_isam2
