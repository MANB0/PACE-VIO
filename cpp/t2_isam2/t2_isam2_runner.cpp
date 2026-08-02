#include "T2Factors.h"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using gtsam::BetweenFactor;
using gtsam::ISAM2;
using gtsam::ISAM2Params;
using gtsam::Matrix;
using gtsam::Matrix6;
using gtsam::Matrix9;
using gtsam::NonlinearFactor;
using gtsam::NonlinearFactorGraph;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Values;
using gtsam::Vector;
using gtsam::Vector3;
using gtsam::Vector6;
using gtsam::Vector9;
using gtsam::imuBias::ConstantBias;
using gtsam::noiseModel::Diagonal;
using gtsam::noiseModel::Gaussian;
using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using t2_isam2::T2CachedImuFactor;
using t2_isam2::T2CompressedVisualFactor;

struct Config {
  fs::path bundle;
  fs::path output;
  double relinearize_threshold = 0.01;
  int relinearize_skip = 1;
  int final_updates = 0;
  double covariance_floor = 1.0e-12;
  double derivative_epsilon = 1.0e-6;
  double audit_absolute_tolerance = 1.0e-5;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
  if (!error.empty()) std::cerr << "error: " << error << "\n\n";
  std::cerr
      << "Usage: " << program << " --bundle DIR --output DIR [options]\n"
      << "  --relinearize-threshold V  iSAM2 threshold (default 0.01)\n"
      << "  --relinearize-skip N       iSAM2 skip (default 1)\n"
      << "  --final-updates N          forced full final updates (default 0)\n"
      << "  --derivative-epsilon V     cached IMU central difference (default 1e-6)\n"
      << "  --audit-abs-tolerance V    cross-language audit limit (default 1e-5)\n";
  std::exit(error.empty() ? 0 : 2);
}

std::string take_value(int& index, int argc, char** argv, const std::string& option) {
  if (++index >= argc) usage(argv[0], "missing value for " + option);
  return argv[index];
}

Config parse_arguments(int argc, char** argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") usage(argv[0]);
    if (arg == "--bundle") config.bundle = take_value(i, argc, argv, arg);
    else if (arg == "--output") config.output = take_value(i, argc, argv, arg);
    else if (arg == "--relinearize-threshold") config.relinearize_threshold = std::stod(take_value(i, argc, argv, arg));
    else if (arg == "--relinearize-skip") config.relinearize_skip = std::stoi(take_value(i, argc, argv, arg));
    else if (arg == "--final-updates") config.final_updates = std::stoi(take_value(i, argc, argv, arg));
    else if (arg == "--derivative-epsilon") config.derivative_epsilon = std::stod(take_value(i, argc, argv, arg));
    else if (arg == "--audit-abs-tolerance") config.audit_absolute_tolerance = std::stod(take_value(i, argc, argv, arg));
    else usage(argv[0], "unknown option " + arg);
  }
  if (config.bundle.empty() || config.output.empty()) {
    usage(argv[0], "--bundle and --output are required");
  }
  if (config.relinearize_skip < 1 || config.final_updates < 0 ||
      !(config.derivative_epsilon > 0.0) || !(config.audit_absolute_tolerance > 0.0)) {
    usage(argv[0], "invalid numerical option");
  }
  return config;
}

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    if (!field.empty() && field.back() == '\r') field.pop_back();
    fields.push_back(field);
  }
  return fields;
}

class CsvTable {
 public:
  explicit CsvTable(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open CSV: " + path.string());
    std::string line;
    if (!std::getline(stream, line)) throw std::runtime_error("empty CSV: " + path.string());
    header_ = split_csv(line);
    for (std::size_t index = 0; index < header_.size(); ++index) {
      if (!column_.emplace(header_[index], index).second) {
        throw std::runtime_error("duplicate CSV column " + header_[index]);
      }
    }
    while (std::getline(stream, line)) {
      if (line.empty()) continue;
      auto fields = split_csv(line);
      if (fields.size() != header_.size()) {
        throw std::runtime_error("CSV row width mismatch in " + path.string());
      }
      rows_.push_back(std::move(fields));
    }
  }

  std::size_t size() const { return rows_.size(); }

  double number(std::size_t row, const std::string& name) const {
    const auto found = column_.find(name);
    if (found == column_.end()) throw std::runtime_error("missing CSV column " + name);
    return std::stod(rows_.at(row).at(found->second));
  }

  std::int64_t integer(std::size_t row, const std::string& name) const {
    const auto found = column_.find(name);
    if (found == column_.end()) throw std::runtime_error("missing CSV column " + name);
    return std::stoll(rows_.at(row).at(found->second));
  }

  Vector vector(std::size_t row, const std::string& prefix, int size) const {
    Vector result(size);
    for (int index = 0; index < size; ++index) {
      result(index) = number(row, prefix + "_" + std::to_string(index));
    }
    return result;
  }

  Matrix matrix(std::size_t row, const std::string& prefix, int rows, int cols) const {
    Matrix result(rows, cols);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        result(r, c) = number(row, prefix + "_" + std::to_string(r) + "_" + std::to_string(c));
      }
    }
    return result;
  }

 private:
  std::vector<std::string> header_;
  std::unordered_map<std::string, std::size_t> column_;
  std::vector<std::vector<std::string>> rows_;
};

template <int N>
Eigen::Matrix<double, N, N> stabilize_covariance(
    const Eigen::Matrix<double, N, N>& covariance, double floor) {
  const auto symmetric = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> solver(symmetric);
  if (solver.info() != Eigen::Success) throw std::runtime_error("covariance eigendecomposition failed");
  auto values = solver.eigenvalues();
  const double scale = std::max(values.cwiseAbs().maxCoeff(), 1.0);
  const double threshold = std::max(floor, std::numeric_limits<double>::epsilon() * scale);
  for (int i = 0; i < N; ++i) values(i) = std::max(values(i), threshold);
  return solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
}

Pose3 pose_from_columns(const CsvTable& table, std::size_t row, const std::string& prefix) {
  const std::string stem = prefix.empty() ? "" : prefix + "_";
  return Pose3(
      Rot3::Quaternion(
          table.number(row, stem + "qw"), table.number(row, stem + "qx"),
          table.number(row, stem + "qy"), table.number(row, stem + "qz")),
      Point3(table.number(row, stem + "tx"), table.number(row, stem + "ty"),
             table.number(row, stem + "tz")));
}

Pose3 pose_from_vector_xyzw(const Vector& value) {
  if (value.size() != 7) throw std::runtime_error("pose vector must have seven entries");
  return Pose3(Rot3::Quaternion(value(6), value(3), value(4), value(5)),
               Point3(value(0), value(1), value(2)));
}

struct StateRecord {
  int local_index{};
  int frame{};
  std::int64_t timestamp_ns{};
  Pose3 pose;
  Vector3 velocity = Vector3::Zero();
  ConstantBias bias;
};

struct EdgeRecord {
  int local_i{};
  int local_j{};
  int frame_i{};
  int frame_j{};
  double dt{};
  bool gravity_in_residual{};
  int visual_rank{};
  Rot3 delta_rotation;
  Vector3 delta_velocity = Vector3::Zero();
  Vector3 delta_position = Vector3::Zero();
  Matrix9 imu_covariance = Matrix9::Identity();
  Eigen::Matrix<double, 9, 6> bias_jacobian = Eigen::Matrix<double, 9, 6>::Zero();
  Vector3 linearized_acc_bias = Vector3::Zero();
  Vector3 linearized_gyro_bias = Vector3::Zero();
  Eigen::Matrix<double, 6, 6> bias_rw_covariance = Eigen::Matrix<double, 6, 6>::Identity();
  Vector3 gravity_world = Vector3::Zero();
  Pose3 visual_reference_CjCi;
  Matrix visual_A;
  Vector visual_c;
  Matrix6 visual_H = Matrix6::Zero();
  Vector6 visual_g = Vector6::Zero();
  Vector expected_visual_residual;
  Matrix expected_visual_J_i;
  Matrix expected_visual_J_j;
  Vector9 expected_imu_residual = Vector9::Zero();
  Matrix expected_imu_J_Xi;
  Matrix expected_imu_J_Vi;
  Matrix expected_imu_J_Bi;
  Matrix expected_imu_J_Xj;
  Matrix expected_imu_J_Vj;
};

std::vector<StateRecord> load_states(const CsvTable& table) {
  std::vector<StateRecord> states;
  states.reserve(table.size());
  for (std::size_t row = 0; row < table.size(); ++row) {
    StateRecord state;
    state.local_index = static_cast<int>(table.integer(row, "local_index"));
    state.frame = static_cast<int>(table.integer(row, "frame"));
    state.timestamp_ns = table.integer(row, "timestamp_ns");
    state.pose = pose_from_columns(table, row, "");
    state.velocity = Vector3(table.number(row, "vx"), table.number(row, "vy"), table.number(row, "vz"));
    state.bias = ConstantBias(
        Vector3(table.number(row, "ba_x"), table.number(row, "ba_y"), table.number(row, "ba_z")),
        Vector3(table.number(row, "bg_x"), table.number(row, "bg_y"), table.number(row, "bg_z")));
    if (state.local_index != static_cast<int>(row)) throw std::runtime_error("state local indices are not contiguous");
    states.push_back(state);
  }
  if (states.size() < 2) throw std::runtime_error("bundle has fewer than two states");
  return states;
}

std::vector<EdgeRecord> load_edges(const CsvTable& table, double covariance_floor) {
  std::vector<EdgeRecord> edges;
  edges.reserve(table.size());
  for (std::size_t row = 0; row < table.size(); ++row) {
    EdgeRecord edge;
    edge.local_i = static_cast<int>(table.integer(row, "local_i"));
    edge.local_j = static_cast<int>(table.integer(row, "local_j"));
    edge.frame_i = static_cast<int>(table.integer(row, "frame_i"));
    edge.frame_j = static_cast<int>(table.integer(row, "frame_j"));
    edge.dt = table.number(row, "dt");
    edge.gravity_in_residual = table.integer(row, "gravity_in_residual") != 0;
    edge.visual_rank = static_cast<int>(table.integer(row, "visual_rank"));
    edge.delta_rotation = Rot3::Expmap(table.vector(row, "delta_rotation_vector", 3));
    edge.delta_velocity = table.vector(row, "delta_velocity", 3);
    edge.delta_position = table.vector(row, "delta_position", 3);
    edge.imu_covariance = stabilize_covariance<9>(table.matrix(row, "imu_covariance", 9, 9), covariance_floor);
    edge.bias_jacobian = table.matrix(row, "bias_jacobian", 9, 6);
    edge.linearized_acc_bias = table.vector(row, "linearized_acc_bias", 3);
    edge.linearized_gyro_bias = table.vector(row, "linearized_gyro_bias", 3);
    edge.bias_rw_covariance = stabilize_covariance<6>(
        table.matrix(row, "bias_rw_covariance", 6, 6), covariance_floor);
    edge.gravity_world = table.vector(row, "gravity_world", 3);
    edge.visual_reference_CjCi = pose_from_vector_xyzw(table.vector(row, "visual_reference_CjCi", 7));
    edge.visual_A = table.matrix(row, "visual_A", 6, 6).topRows(edge.visual_rank);
    edge.visual_c = table.vector(row, "visual_c", 6).head(edge.visual_rank);
    edge.visual_H = table.matrix(row, "visual_H", 6, 6);
    edge.visual_g = table.vector(row, "visual_g", 6);
    edge.expected_visual_residual = table.vector(row, "expected_visual_residual", 6).head(edge.visual_rank);
    edge.expected_visual_J_i = table.matrix(row, "expected_visual_J_i", 6, 6).topRows(edge.visual_rank);
    edge.expected_visual_J_j = table.matrix(row, "expected_visual_J_j", 6, 6).topRows(edge.visual_rank);
    edge.expected_imu_residual = table.vector(row, "expected_imu_residual", 9);
    edge.expected_imu_J_Xi = table.matrix(row, "expected_imu_J_Xi", 9, 6);
    edge.expected_imu_J_Vi = table.matrix(row, "expected_imu_J_Vi", 9, 3);
    edge.expected_imu_J_Bi = table.matrix(row, "expected_imu_J_Bi", 9, 6);
    edge.expected_imu_J_Xj = table.matrix(row, "expected_imu_J_Xj", 9, 6);
    edge.expected_imu_J_Vj = table.matrix(row, "expected_imu_J_Vj", 9, 3);
    if (edge.local_i != static_cast<int>(row) || edge.local_j != edge.local_i + 1) {
      throw std::runtime_error("edge local indices are not contiguous");
    }
    edges.push_back(edge);
  }
  return edges;
}

double max_abs(const Matrix& matrix) {
  return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

struct AuditSummary {
  double max_h_error{};
  double max_g_error{};
  double max_visual_residual_error{};
  double max_visual_jacobian_error{};
  double max_imu_residual_error{};
  double max_imu_jacobian_error{};
  bool finite = true;
};

AuditSummary audit_factors(
    const std::vector<StateRecord>& states, const std::vector<EdgeRecord>& edges,
    const Pose3& extrinsic_CI, double derivative_epsilon) {
  AuditSummary summary;
  for (const auto& edge : edges) {
    const auto& state_i = states.at(edge.local_i);
    const auto& state_j = states.at(edge.local_j);
    auto visual = std::make_shared<T2CompressedVisualFactor>(
        X(edge.local_i), X(edge.local_j), edge.visual_reference_CjCi,
        edge.visual_A, edge.visual_c, extrinsic_CI);
    Matrix H_vi, H_vj;
    const Vector visual_residual = visual->evaluateError(
        state_i.pose, state_j.pose, H_vi, H_vj);
    summary.max_h_error = std::max(summary.max_h_error,
        max_abs(edge.visual_A.transpose() * edge.visual_A - edge.visual_H));
    summary.max_g_error = std::max(summary.max_g_error,
        max_abs(edge.visual_A.transpose() * edge.visual_c - edge.visual_g));
    summary.max_visual_residual_error = std::max(summary.max_visual_residual_error,
        max_abs(visual_residual - edge.expected_visual_residual));
    summary.max_visual_jacobian_error = std::max({
        summary.max_visual_jacobian_error,
        max_abs(H_vi - edge.expected_visual_J_i),
        max_abs(H_vj - edge.expected_visual_J_j)});

    auto imu = std::make_shared<T2CachedImuFactor>(
        X(edge.local_i), V(edge.local_i), B(edge.local_i),
        X(edge.local_j), V(edge.local_j), edge.delta_rotation,
        edge.delta_velocity, edge.delta_position, edge.dt,
        edge.imu_covariance, edge.bias_jacobian,
        edge.linearized_acc_bias, edge.linearized_gyro_bias,
        edge.gravity_world, edge.gravity_in_residual, derivative_epsilon);
    Matrix H_xi, H_vi_imu, H_bi, H_xj, H_vj_imu;
    const Vector imu_residual = imu->evaluateError(
        state_i.pose, state_i.velocity, state_i.bias,
        state_j.pose, state_j.velocity,
        H_xi, H_vi_imu, H_bi, H_xj, H_vj_imu);
    summary.max_imu_residual_error = std::max(summary.max_imu_residual_error,
        max_abs(imu_residual - edge.expected_imu_residual));
    summary.max_imu_jacobian_error = std::max({
        summary.max_imu_jacobian_error,
        max_abs(H_xi - edge.expected_imu_J_Xi),
        max_abs(H_vi_imu - edge.expected_imu_J_Vi),
        max_abs(H_bi - edge.expected_imu_J_Bi),
        max_abs(H_xj - edge.expected_imu_J_Xj),
        max_abs(H_vj_imu - edge.expected_imu_J_Vj)});
    summary.finite = summary.finite && visual_residual.allFinite() &&
        H_vi.allFinite() && H_vj.allFinite() && imu_residual.allFinite() &&
        H_xi.allFinite() && H_vi_imu.allFinite() && H_bi.allFinite() &&
        H_xj.allFinite() && H_vj_imu.allFinite();
  }
  return summary;
}

void write_audit(const fs::path& path, const AuditSummary& audit, std::size_t edge_count) {
  std::ofstream stream(path);
  stream << std::setprecision(17)
         << "{\n"
         << "  \"edge_count\": " << edge_count << ",\n"
         << "  \"max_hessian_absolute_error\": " << audit.max_h_error << ",\n"
         << "  \"max_gradient_absolute_error\": " << audit.max_g_error << ",\n"
         << "  \"max_visual_residual_absolute_error\": " << audit.max_visual_residual_error << ",\n"
         << "  \"max_visual_jacobian_absolute_error\": " << audit.max_visual_jacobian_error << ",\n"
         << "  \"max_imu_residual_absolute_error\": " << audit.max_imu_residual_error << ",\n"
         << "  \"max_imu_jacobian_absolute_error\": " << audit.max_imu_jacobian_error << ",\n"
         << "  \"has_nan_or_inf\": " << (audit.finite ? "false" : "true") << "\n"
         << "}\n";
}

struct SolveSummary {
  double elapsed_s{};
  double update_median_ms{};
  double update_p95_ms{};
  double update_max_ms{};
  double initial_prior_cost{};
  double imu_cost{};
  double bias_cost{};
  double visual_cost{};
};

double percentile(std::vector<double> values, double probability) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double location = probability * static_cast<double>(values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(location));
  const std::size_t upper = std::min(lower + 1, values.size() - 1);
  const double alpha = location - static_cast<double>(lower);
  return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

SolveSummary solve_incrementally(
    const Config& config, const std::vector<StateRecord>& states,
    const std::vector<EdgeRecord>& edges, const Pose3& extrinsic_CI,
    const Vector& prior_sigma_t2) {
  ISAM2Params params;
  params.relinearizeThreshold = config.relinearize_threshold;
  params.relinearizeSkip = config.relinearize_skip;
  params.factorization = ISAM2Params::CHOLESKY;
  ISAM2 isam(params);

  const Matrix6 permutation = t2_isam2::TranslationRotationFromRotationTranslation();
  const Vector6 pose_sigma_rt = permutation * prior_sigma_t2.head<6>();
  const auto pose_prior_noise = Diagonal::Sigmas(pose_sigma_rt);
  const auto velocity_prior_noise = Diagonal::Sigmas(prior_sigma_t2.segment<3>(6));
  const auto bias_prior_noise = Diagonal::Sigmas(prior_sigma_t2.segment<6>(9));

  auto pose_prior = std::make_shared<gtsam::PriorFactor<Pose3>>(
      X(0), states.front().pose, pose_prior_noise);
  auto velocity_prior = std::make_shared<gtsam::PriorFactor<Vector3>>(
      V(0), states.front().velocity, velocity_prior_noise);
  auto bias_prior = std::make_shared<gtsam::PriorFactor<ConstantBias>>(
      B(0), states.front().bias, bias_prior_noise);

  NonlinearFactorGraph new_factors;
  Values new_values;
  new_factors.push_back(pose_prior);
  new_factors.push_back(velocity_prior);
  new_factors.push_back(bias_prior);
  new_values.insert(X(0), states.front().pose);
  new_values.insert(V(0), states.front().velocity);
  new_values.insert(B(0), states.front().bias);
  isam.update(new_factors, new_values);
  new_factors.resize(0);
  new_values.clear();

  std::vector<std::shared_ptr<T2CachedImuFactor>> imu_factors;
  std::vector<std::shared_ptr<BetweenFactor<ConstantBias>>> bias_factors;
  std::vector<std::shared_ptr<T2CompressedVisualFactor>> visual_factors;
  std::vector<double> update_ms;
  std::ofstream timing(config.output / "isam2_update_timing.csv");
  timing << "local_j,frame_j,update_ms\n";

  const auto started = std::chrono::steady_clock::now();
  for (const auto& edge : edges) {
    auto imu = std::make_shared<T2CachedImuFactor>(
        X(edge.local_i), V(edge.local_i), B(edge.local_i),
        X(edge.local_j), V(edge.local_j), edge.delta_rotation,
        edge.delta_velocity, edge.delta_position, edge.dt,
        edge.imu_covariance, edge.bias_jacobian,
        edge.linearized_acc_bias, edge.linearized_gyro_bias,
        edge.gravity_world, edge.gravity_in_residual,
        config.derivative_epsilon);
    auto bias = std::make_shared<BetweenFactor<ConstantBias>>(
        B(edge.local_i), B(edge.local_j), ConstantBias(),
        Gaussian::Covariance(edge.bias_rw_covariance));
    auto visual = std::make_shared<T2CompressedVisualFactor>(
        X(edge.local_i), X(edge.local_j), edge.visual_reference_CjCi,
        edge.visual_A, edge.visual_c, extrinsic_CI);
    imu_factors.push_back(imu);
    bias_factors.push_back(bias);
    visual_factors.push_back(visual);
    new_factors.push_back(imu);
    new_factors.push_back(bias);
    new_factors.push_back(visual);

    const auto& state_j = states.at(edge.local_j);
    new_values.insert(X(edge.local_j), state_j.pose);
    new_values.insert(V(edge.local_j), state_j.velocity);
    new_values.insert(B(edge.local_j), state_j.bias);
    const auto before = std::chrono::steady_clock::now();
    isam.update(new_factors, new_values);
    const auto after = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(after - before).count();
    update_ms.push_back(milliseconds);
    timing << edge.local_j << ',' << edge.frame_j << ',' << std::setprecision(17) << milliseconds << '\n';
    new_factors.resize(0);
    new_values.clear();
  }
  for (int update = 0; update < config.final_updates; ++update) {
    gtsam::ISAM2UpdateParams update_params;
    update_params.force_relinearize = true;
    update_params.forceFullSolve = true;
    isam.update(NonlinearFactorGraph(), Values(), update_params);
  }
  const Values estimate = isam.calculateEstimate();
  const double elapsed_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();

  std::ofstream states_output(config.output / "isam2_states_internal.csv");
  states_output << std::setprecision(17)
      << "local_index,frame,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,vx,vy,vz,ba_x,ba_y,ba_z,bg_x,bg_y,bg_z\n";
  for (const auto& record : states) {
    const Pose3 pose = estimate.at<Pose3>(X(record.local_index));
    const Vector3 velocity = estimate.at<Vector3>(V(record.local_index));
    const ConstantBias bias = estimate.at<ConstantBias>(B(record.local_index));
    const auto quaternion = pose.rotation().toQuaternion();
    states_output << record.local_index << ',' << record.frame << ',' << record.timestamp_ns << ','
        << pose.x() << ',' << pose.y() << ',' << pose.z() << ','
        << quaternion.x() << ',' << quaternion.y() << ',' << quaternion.z() << ',' << quaternion.w() << ','
        << velocity.x() << ',' << velocity.y() << ',' << velocity.z() << ','
        << bias.accelerometer().x() << ',' << bias.accelerometer().y() << ',' << bias.accelerometer().z() << ','
        << bias.gyroscope().x() << ',' << bias.gyroscope().y() << ',' << bias.gyroscope().z() << '\n';
  }

  SolveSummary summary;
  summary.elapsed_s = elapsed_s;
  summary.update_median_ms = percentile(update_ms, 0.5);
  summary.update_p95_ms = percentile(update_ms, 0.95);
  summary.update_max_ms = update_ms.empty() ? 0.0 : *std::max_element(update_ms.begin(), update_ms.end());
  summary.initial_prior_cost = pose_prior->error(estimate) + velocity_prior->error(estimate) + bias_prior->error(estimate);
  for (std::size_t index = 0; index < edges.size(); ++index) {
    summary.imu_cost += imu_factors[index]->error(estimate);
    summary.bias_cost += bias_factors[index]->error(estimate);
    summary.visual_cost += visual_factors[index]->error(estimate);
  }
  return summary;
}

void write_solve_summary(const fs::path& path, const SolveSummary& summary,
                         std::size_t state_count, std::size_t edge_count) {
  std::ofstream stream(path);
  stream << std::setprecision(17)
      << "{\n"
      << "  \"backend\": \"PACE compressed UVD + cached IMU + iSAM2\",\n"
      << "  \"state_count\": " << state_count << ",\n"
      << "  \"edge_count\": " << edge_count << ",\n"
      << "  \"elapsed_s\": " << summary.elapsed_s << ",\n"
      << "  \"update_median_ms\": " << summary.update_median_ms << ",\n"
      << "  \"update_p95_ms\": " << summary.update_p95_ms << ",\n"
      << "  \"update_max_ms\": " << summary.update_max_ms << ",\n"
      << "  \"factor_cost\": {\n"
      << "    \"prior\": " << summary.initial_prior_cost << ",\n"
      << "    \"imu\": " << summary.imu_cost << ",\n"
      << "    \"bias\": " << summary.bias_cost << ",\n"
      << "    \"visual\": " << summary.visual_cost << "\n"
      << "  }\n"
      << "}\n";
}

int run(const Config& config) {
  fs::create_directories(config.output);
  const CsvTable state_table(config.bundle / "states.csv");
  const CsvTable edge_table(config.bundle / "edges.csv");
  const CsvTable contract_table(config.bundle / "contract.csv");
  const auto states = load_states(state_table);
  const auto edges = load_edges(edge_table, config.covariance_floor);
  if (edges.size() + 1 != states.size()) throw std::runtime_error("state/edge count mismatch");

  Vector extrinsic_vector(7);
  const std::vector<std::string> extrinsic_names = {
      "extrinsic_tx", "extrinsic_ty", "extrinsic_tz", "extrinsic_qx",
      "extrinsic_qy", "extrinsic_qz", "extrinsic_qw"};
  for (int index = 0; index < 7; ++index) extrinsic_vector(index) = contract_table.number(0, extrinsic_names[index]);
  const Pose3 extrinsic_CI = pose_from_vector_xyzw(extrinsic_vector);
  Vector prior_sigma(15);
  for (int index = 0; index < 15; ++index) {
    prior_sigma(index) = contract_table.number(0, "prior_sigma_" + std::to_string(index));
  }

  const AuditSummary audit = audit_factors(
      states, edges, extrinsic_CI, config.derivative_epsilon);
  write_audit(config.output / "factor_equivalence_summary.json", audit, edges.size());
  const double worst_audit = std::max({
      audit.max_h_error, audit.max_g_error, audit.max_visual_residual_error,
      audit.max_visual_jacobian_error, audit.max_imu_residual_error,
      audit.max_imu_jacobian_error});
  if (!audit.finite || worst_audit > config.audit_absolute_tolerance) {
    throw std::runtime_error("PACE/GTSAM cross-language factor audit failed; inspect factor_equivalence_summary.json");
  }

  const SolveSummary solve = solve_incrementally(
      config, states, edges, extrinsic_CI, prior_sigma);
  write_solve_summary(config.output / "isam2_summary.json", solve, states.size(), edges.size());
  std::cout << std::setprecision(10)
      << "PACE-VIO iSAM2 completed " << edges.size() << " edges\n"
      << "  factor audit max abs: " << worst_audit << "\n"
      << "  update median/P95/max: " << solve.update_median_ms << " / "
      << solve.update_p95_ms << " / " << solve.update_max_ms << " ms\n"
      << "  factor cost prior/imu/bias/visual: " << solve.initial_prior_cost << " / "
      << solve.imu_cost << " / " << solve.bias_cost << " / " << solve.visual_cost << "\n";
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(parse_arguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "fatal: " << error.what() << '\n';
    return 1;
  }
}
