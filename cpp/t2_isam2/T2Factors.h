#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

#include <cmath>
#include <stdexcept>

namespace t2_isam2 {

using gtsam::Matrix;
using gtsam::Matrix6;
using gtsam::Matrix9;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::Vector6;
using gtsam::Vector9;
using gtsam::imuBias::ConstantBias;

inline Matrix6 TranslationRotationFromRotationTranslation() {
  Matrix6 permutation = Matrix6::Zero();
  permutation.block<3, 3>(0, 3) = gtsam::Matrix3::Identity();
  permutation.block<3, 3>(3, 0) = gtsam::Matrix3::Identity();
  return permutation;
}

class T2CompressedVisualFactor
    : public gtsam::NoiseModelFactorN<Pose3, Pose3> {
 public:
  using Base = gtsam::NoiseModelFactorN<Pose3, Pose3>;
  using Base::evaluateError;

  T2CompressedVisualFactor(
      gtsam::Key pose_i_key, gtsam::Key pose_j_key,
      const Pose3& reference_CjCi, const Matrix& sqrt_information_tr,
      const Vector& residual_offset, const Pose3& extrinsic_CI)
      : Base(gtsam::noiseModel::Unit::Create(residual_offset.size()),
             pose_i_key, pose_j_key),
        reference_CjCi_(reference_CjCi),
        sqrt_information_tr_(sqrt_information_tr),
        residual_offset_(residual_offset),
        extrinsic_CI_(extrinsic_CI) {
    if (sqrt_information_tr_.cols() != 6 ||
        sqrt_information_tr_.rows() != residual_offset_.size() ||
        residual_offset_.size() < 1 || residual_offset_.size() > 6) {
      throw std::invalid_argument("invalid compressed T2 visual factor shape");
    }
    if (!sqrt_information_tr_.allFinite() || !residual_offset_.allFinite()) {
      throw std::invalid_argument("compressed T2 visual factor contains NaN/Inf");
    }
  }

  Vector evaluateError(
      const Pose3& pose_WI_i, const Pose3& pose_WI_j,
      gtsam::OptionalMatrixType H_i = nullptr,
      gtsam::OptionalMatrixType H_j = nullptr) const override {
    Matrix6 H_WCi_WIi, H_WCj_WIj;
    const Pose3 inverse_extrinsic = extrinsic_CI_.inverse();
    const Pose3 pose_WC_i = pose_WI_i.compose(
        inverse_extrinsic, (H_i || H_j) ? &H_WCi_WIi : nullptr);
    const Pose3 pose_WC_j = pose_WI_j.compose(
        inverse_extrinsic, (H_i || H_j) ? &H_WCj_WIj : nullptr);

    Matrix6 H_current_WCj, H_current_WCi;
    const Pose3 current_CjCi = pose_WC_j.between(
        pose_WC_i,
        (H_i || H_j) ? &H_current_WCj : nullptr,
        (H_i || H_j) ? &H_current_WCi : nullptr);

    Matrix6 H_local_current;
    const Pose3 local_pose = reference_CjCi_.between(
        current_CjCi, nullptr, (H_i || H_j) ? &H_local_current : nullptr);
    Matrix6 H_log_local;
    const Vector6 local_rt = Pose3::Logmap(
        local_pose, (H_i || H_j) ? &H_log_local : nullptr);
    const Matrix6 permutation = TranslationRotationFromRotationTranslation();
    const Vector6 local_tr = permutation * local_rt;

    if (H_i) {
      *H_i = sqrt_information_tr_ * permutation * H_log_local *
             H_local_current * H_current_WCi * H_WCi_WIi;
    }
    if (H_j) {
      *H_j = sqrt_information_tr_ * permutation * H_log_local *
             H_local_current * H_current_WCj * H_WCj_WIj;
    }
    return sqrt_information_tr_ * local_tr + residual_offset_;
  }

  const Matrix& sqrtInformationTR() const { return sqrt_information_tr_; }
  const Vector& residualOffset() const { return residual_offset_; }

 private:
  Pose3 reference_CjCi_;
  Matrix sqrt_information_tr_;
  Vector residual_offset_;
  Pose3 extrinsic_CI_;
};

class T2CachedImuFactor
    : public gtsam::NoiseModelFactorN<
          Pose3, Vector3, ConstantBias, Pose3, Vector3> {
 public:
  using Base = gtsam::NoiseModelFactorN<
      Pose3, Vector3, ConstantBias, Pose3, Vector3>;
  using Base::evaluateError;

  T2CachedImuFactor(
      gtsam::Key pose_i_key, gtsam::Key velocity_i_key,
      gtsam::Key bias_i_key, gtsam::Key pose_j_key,
      gtsam::Key velocity_j_key, const Rot3& delta_rotation,
      const Vector3& delta_velocity, const Vector3& delta_position,
      double dt, const Matrix9& covariance_pvr,
      const Eigen::Matrix<double, 9, 6>& bias_jacobian,
      const Vector3& linearized_acc_bias,
      const Vector3& linearized_gyro_bias,
      const Vector3& gravity_world, bool gravity_in_residual,
      double derivative_epsilon = 1.0e-6)
      : Base(gtsam::noiseModel::Gaussian::Covariance(covariance_pvr),
             pose_i_key, velocity_i_key, bias_i_key,
             pose_j_key, velocity_j_key),
        delta_rotation_(delta_rotation),
        delta_velocity_(delta_velocity),
        delta_position_(delta_position),
        dt_(dt),
        bias_jacobian_(bias_jacobian),
        linearized_acc_bias_(linearized_acc_bias),
        linearized_gyro_bias_(linearized_gyro_bias),
        gravity_world_(gravity_world),
        gravity_in_residual_(gravity_in_residual),
        derivative_epsilon_(derivative_epsilon) {
    if (!(dt_ > 0.0) || !std::isfinite(dt_)) {
      throw std::invalid_argument("T2 cached IMU factor has invalid dt");
    }
    if (!covariance_pvr.allFinite() || !bias_jacobian_.allFinite()) {
      throw std::invalid_argument("T2 cached IMU factor contains NaN/Inf");
    }
  }

  Vector evaluateError(
      const Pose3& pose_i, const Vector3& velocity_i,
      const ConstantBias& bias_i, const Pose3& pose_j,
      const Vector3& velocity_j,
      gtsam::OptionalMatrixType H_pose_i = nullptr,
      gtsam::OptionalMatrixType H_velocity_i = nullptr,
      gtsam::OptionalMatrixType H_bias_i = nullptr,
      gtsam::OptionalMatrixType H_pose_j = nullptr,
      gtsam::OptionalMatrixType H_velocity_j = nullptr) const override {
    Vector6 delta_bias;
    delta_bias.head<3>() = bias_i.accelerometer() - linearized_acc_bias_;
    delta_bias.tail<3>() = bias_i.gyroscope() - linearized_gyro_bias_;
    const Vector9 correction = bias_jacobian_ * delta_bias;

    const Vector3 corrected_position = delta_position_ + correction.segment<3>(0);
    const Vector3 corrected_velocity = delta_velocity_ + correction.segment<3>(3);

    gtsam::Matrix3 H_exp_correction;
    const Rot3 correction_rotation = Rot3::Expmap(
        correction.segment<3>(6), H_exp_correction);
    gtsam::Matrix3 H_corrected_correction;
    const Rot3 corrected_rotation = delta_rotation_.compose(
        correction_rotation, nullptr, H_corrected_correction);

    const Rot3 rotation_i = pose_i.rotation();
    const Rot3 rotation_j = pose_j.rotation();
    gtsam::Matrix3 H_relative_rotation_i, H_relative_rotation_j;
    const Rot3 relative_rotation = rotation_i.between(
        rotation_j, H_relative_rotation_i, H_relative_rotation_j);

    gtsam::Matrix3 H_rotation_error_corrected, H_rotation_error_relative;
    const Rot3 rotation_error = corrected_rotation.between(
        relative_rotation,
        H_rotation_error_corrected,
        H_rotation_error_relative);
    gtsam::Matrix3 H_log_error;
    const Vector3 rotation_residual = Rot3::Logmap(
        rotation_error, H_log_error);

    const Pose3 relative_ij = pose_i.between(pose_j);
    const Vector3 gravity_body = gravity_in_residual_
        ? rotation_i.unrotate(gravity_world_)
        : Vector3::Zero();
    const Vector3 velocity_i_body = rotation_i.unrotate(velocity_i);
    const Vector3 delta_velocity_body = rotation_i.unrotate(
        velocity_j - velocity_i);
    const Vector3 position_kinematic =
        relative_ij.translation() - velocity_i_body * dt_ -
        0.5 * gravity_body * dt_ * dt_;
    const Vector3 velocity_kinematic =
        delta_velocity_body - gravity_body * dt_;

    Vector9 residual;
    residual.segment<3>(0) = position_kinematic - corrected_position;
    residual.segment<3>(3) = velocity_kinematic - corrected_velocity;
    residual.segment<3>(6) = rotation_residual;

    if (H_pose_i) {
      Matrix H = Matrix::Zero(9, 6);
      H.block<3, 3>(0, 0) = gtsam::skewSymmetric(position_kinematic);
      H.block<3, 3>(0, 3) = -gtsam::Matrix3::Identity();
      H.block<3, 3>(3, 0) = gtsam::skewSymmetric(velocity_kinematic);
      H.block<3, 3>(6, 0) = H_log_error * H_rotation_error_relative *
                             H_relative_rotation_i;
      *H_pose_i = H;
    }
    if (H_velocity_i) {
      Matrix H = Matrix::Zero(9, 3);
      H.block<3, 3>(0, 0) = -rotation_i.matrix().transpose() * dt_;
      H.block<3, 3>(3, 0) = -rotation_i.matrix().transpose();
      *H_velocity_i = H;
    }
    if (H_bias_i) {
      Matrix H = Matrix::Zero(9, 6);
      H.block<6, 6>(0, 0) = -bias_jacobian_.topRows<6>();
      H.block<3, 6>(6, 0) =
          H_log_error * H_rotation_error_corrected *
          H_corrected_correction * H_exp_correction *
          bias_jacobian_.bottomRows<3>();
      *H_bias_i = H;
    }
    if (H_pose_j) {
      Matrix H = Matrix::Zero(9, 6);
      H.block<3, 3>(0, 3) = relative_rotation.matrix();
      H.block<3, 3>(6, 0) = H_log_error * H_rotation_error_relative *
                             H_relative_rotation_j;
      *H_pose_j = H;
    }
    if (H_velocity_j) {
      Matrix H = Matrix::Zero(9, 3);
      H.block<3, 3>(3, 0) = rotation_i.matrix().transpose();
      *H_velocity_j = H;
    }
    return residual;
  }

  Vector9 unwhitenedResidual(
      const Pose3& pose_i, const Vector3& velocity_i,
      const ConstantBias& bias_i, const Pose3& pose_j,
      const Vector3& velocity_j) const {
    Vector6 delta_bias;
    delta_bias.head<3>() = bias_i.accelerometer() - linearized_acc_bias_;
    delta_bias.tail<3>() = bias_i.gyroscope() - linearized_gyro_bias_;
    const Vector9 correction = bias_jacobian_ * delta_bias;

    const Vector3 corrected_position = delta_position_ + correction.segment<3>(0);
    const Vector3 corrected_velocity = delta_velocity_ + correction.segment<3>(3);
    const Rot3 corrected_rotation = delta_rotation_.compose(
        Rot3::Expmap(correction.segment<3>(6)));

    const Pose3 relative_ij = pose_i.between(pose_j);
    const Vector3 gravity_body = gravity_in_residual_
        ? pose_i.rotation().unrotate(gravity_world_)
        : Vector3::Zero();
    const Vector3 velocity_i_body = pose_i.rotation().unrotate(velocity_i);
    const Vector3 delta_velocity_body = pose_i.rotation().unrotate(
        velocity_j - velocity_i);
    const Rot3 relative_rotation = pose_i.rotation().between(pose_j.rotation());

    Vector9 residual;
    residual.segment<3>(0) = relative_ij.translation() - velocity_i_body * dt_ -
        0.5 * gravity_body * dt_ * dt_ - corrected_position;
    residual.segment<3>(3) = delta_velocity_body - gravity_body * dt_ -
        corrected_velocity;
    residual.segment<3>(6) = Rot3::Logmap(
        corrected_rotation.between(relative_rotation));
    return residual;
  }

 private:
  Rot3 delta_rotation_;
  Vector3 delta_velocity_;
  Vector3 delta_position_;
  double dt_;
  Eigen::Matrix<double, 9, 6> bias_jacobian_;
  Vector3 linearized_acc_bias_;
  Vector3 linearized_gyro_bias_;
  Vector3 gravity_world_;
  bool gravity_in_residual_;
  double derivative_epsilon_;
};

}  // namespace t2_isam2
