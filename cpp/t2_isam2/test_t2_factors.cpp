#include "T2Factors.h"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using gtsam::Matrix;
using gtsam::Matrix6;
using gtsam::Matrix9;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::Vector6;
using gtsam::imuBias::ConstantBias;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using t2_isam2::T2CachedImuFactor;
using t2_isam2::T2CompressedVisualFactor;

double max_abs(const Matrix& matrix) {
  return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void test_visual_jacobian() {
  const Pose3 pose_i(Rot3::RzRyRx(0.13, -0.09, 0.21), Point3(0.4, -0.2, 0.7));
  const Pose3 pose_j(Rot3::RzRyRx(-0.08, 0.16, 0.31), Point3(0.9, 0.1, 0.5));
  const Pose3 extrinsic(Rot3::RzRyRx(0.03, -0.02, 0.01), Point3(0.18, -0.04, 0.09));
  const Pose3 current = pose_j.compose(extrinsic.inverse()).between(
      pose_i.compose(extrinsic.inverse()));
  const Pose3 reference = current.compose(Pose3::Expmap(
      (Vector6() << 0.01, -0.02, 0.015, 0.03, -0.01, 0.02).finished()));
  Matrix6 A = Matrix6::Identity();
  A.diagonal() << 2.0, 3.0, 4.0, 5.0, 6.0, 7.0;
  const Vector6 c = (Vector6() << 0.2, -0.1, 0.05, 0.03, -0.02, 0.01).finished();
  T2CompressedVisualFactor factor(X(0), X(1), reference, A, c, extrinsic);

  Matrix H_i, H_j;
  factor.evaluateError(pose_i, pose_j, H_i, H_j);
  const auto function = [&factor](const Pose3& first, const Pose3& second) {
    return factor.evaluateError(first, second);
  };
  const Matrix numerical_i = gtsam::numericalDerivative21(
      function, pose_i, pose_j, 1.0e-6);
  const Matrix numerical_j = gtsam::numericalDerivative22(
      function, pose_i, pose_j, 1.0e-6);
  const double error_i = max_abs(H_i - numerical_i);
  const double error_j = max_abs(H_j - numerical_j);
  std::cout << "visual Jacobian max abs i/j: " << error_i << " / " << error_j << '\n';
  require(error_i < 1.0e-6 && error_j < 1.0e-6,
          "compressed visual analytic Jacobian differs from central difference");
}

void test_imu_residual_and_zero_blocks() {
  const Pose3 pose_i(Rot3::RzRyRx(0.12, -0.07, 0.18), Point3(0.3, -0.4, 0.8));
  const Rot3 delta_rotation = Rot3::Expmap(Vector3(0.02, -0.01, 0.03));
  const Pose3 relative(delta_rotation, Point3(0.07, -0.02, 0.01));
  const Pose3 pose_j = pose_i.compose(relative);
  const Vector3 velocity_i(0.4, -0.1, 0.2);
  const Vector3 velocity_j(0.43, -0.08, 0.19);
  const ConstantBias bias(Vector3(0.01, -0.02, 0.03), Vector3(0.001, -0.002, 0.003));
  const double dt = 0.1;
  const Vector3 gravity_world = Vector3::Zero();
  const Vector3 delta_p = relative.translation() - pose_i.rotation().unrotate(velocity_i) * dt;
  const Vector3 delta_v = pose_i.rotation().unrotate(velocity_j - velocity_i);
  const Matrix9 covariance = Matrix9::Identity();
  const Eigen::Matrix<double, 9, 6> bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
  T2CachedImuFactor factor(
      X(0), V(0), B(0), X(1), V(1), delta_rotation, delta_v, delta_p,
      dt, covariance, bias_jacobian, bias.accelerometer(), bias.gyroscope(),
      gravity_world, true, 1.0e-6);

  Matrix H_xi, H_vi, H_bi, H_xj, H_vj;
  const Vector residual = factor.evaluateError(
      pose_i, velocity_i, bias, pose_j, velocity_j,
      H_xi, H_vi, H_bi, H_xj, H_vj);
  require(max_abs(residual) < 1.0e-10, "synthetic cached IMU residual does not close");
  require(H_xi.allFinite() && H_vi.allFinite() && H_bi.allFinite() &&
          H_xj.allFinite() && H_vj.allFinite(), "cached IMU Jacobian contains NaN/Inf");

  const double rp_phi_j = max_abs(H_xj.block(0, 0, 3, 3));
  const double rv_pose_j = max_abs(H_xj.block(3, 0, 3, 6));
  const double rR_translation_j = max_abs(H_xj.block(6, 3, 3, 3));
  std::cout << "IMU theoretical zero max abs rp/phi_j, rv/pose_j, rR/t_j: "
            << rp_phi_j << " / " << rv_pose_j << " / " << rR_translation_j << '\n';
  require(rp_phi_j < 1.0e-8 && rv_pose_j < 1.0e-8 && rR_translation_j < 1.0e-8,
          "cached IMU theoretical zero block is nonzero");
}

int main() {
  try {
    test_visual_jacobian();
    test_imu_residual_and_zero_blocks();
    std::cout << "T2-iSAM2 factor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
