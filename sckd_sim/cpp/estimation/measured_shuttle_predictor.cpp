#include "estimation/measured_shuttle_predictor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ball-mlp/ball_mlp.h"

namespace sckd {
namespace {

struct SourceRow {
  double timestamp_ms = 0.0;
  std::array<double, 3> position_cm{};
};

void RequireFinite(double value, const char* name) {
  if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " must be finite");
}

}  // namespace

struct MeasuredShuttlePredictor::Impl {
  explicit Impl(Config value) : config(std::move(value)) {
    ValidateConfig();
    ReadRows();
    stats.input_policy =
        config.preassociated_replay ? "preassociated_forced" : "normal_association";

    auto params = ball::mlp::FramesFillerParams::createFps1000FillerParams();
    params.common_params.track_time_source = ball::mlp::TrackTimeSource::LatestNodeTime;
    // Time validity follows the recorded capture clock, not wall time.
    params.common_params.check_timeout = true;
    predictor = std::make_unique<ball::mlp::BallMLP>(params);
    predictor->loadNet20Model(config.net20_model_path, config.net20_normalizer_path);
    predictor->setTrackTimeSource(ball::mlp::TrackTimeSource::LatestNodeTime);
    track = predictor->createSingleTrack();
  }

  void ValidateConfig() const {
    if (!std::filesystem::is_regular_file(config.trajectory_path))
      throw std::invalid_argument("shuttle trajectory not found: " + config.trajectory_path);
    if (!std::filesystem::is_regular_file(config.net20_model_path))
      throw std::invalid_argument("ball-mlp Net20 model not found: " + config.net20_model_path);
    if (!std::filesystem::is_regular_file(config.net20_normalizer_path))
      throw std::invalid_argument("ball-mlp normalizer not found: " + config.net20_normalizer_path);
    RequireFinite(config.fixed_height_m, "fixed_height_m");
    RequireFinite(config.height_tolerance_m, "height_tolerance_m");
    RequireFinite(config.prediction_stride_ms, "prediction_stride_ms");
    RequireFinite(config.initial_estimated_hit_s, "initial_estimated_hit_s");
    if (config.fixed_height_m <= config.height_tolerance_m || config.height_tolerance_m <= 0.0 ||
        config.prediction_stride_ms <= 0.0 || config.max_prediction_samples == 0 ||
        config.initial_estimated_hit_s <= 0.0)
      throw std::invalid_argument("shuttle predictor config values must be positive");
  }

  void ReadRows() {
    std::ifstream input(config.trajectory_path);
    if (!input) throw std::runtime_error("cannot open shuttle trajectory: " + config.trajectory_path);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      if (line.empty()) continue;
      std::istringstream fields(line);
      SourceRow row;
      if (!(fields >> row.timestamp_ms >> row.position_cm[0] >> row.position_cm[1] >>
            row.position_cm[2]))
        throw std::runtime_error("invalid shuttle trajectory row " + std::to_string(line_number));
      if (!std::isfinite(row.timestamp_ms) ||
          !std::all_of(row.position_cm.begin(), row.position_cm.end(),
                       [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("nonfinite shuttle trajectory row " + std::to_string(line_number));
      if (!rows.empty() && row.timestamp_ms <= rows.back().timestamp_ms)
        throw std::runtime_error("non-increasing shuttle timestamp at row " +
                                 std::to_string(line_number));
      rows.push_back(row);
    }
    if (rows.size() < 2) throw std::runtime_error("shuttle trajectory requires at least two rows");
    first_timestamp_ms = rows.front().timestamp_ms;
  }

  float EstimatedHitTimeMs(const SourceRow& row) const {
    if (last_valid_hit_time_s) {
      const double absolute_hit_ms = first_timestamp_ms + 1000.0 * *last_valid_hit_time_s;
      return static_cast<float>(std::max(0.0, absolute_hit_ms - row.timestamp_ms));
    }
    return static_cast<float>(1000.0 * config.initial_estimated_hit_s);
  }

  bool PredictedVelocityMps(double timestamp_ms, std::array<double, 3>& velocity_m_s) const {
    const auto& samples = track->getRpts();
    if (samples.size() < 2) return false;

    const auto at_or_after =
        std::lower_bound(samples.begin(), samples.end(), timestamp_ms,
                         [](const ball::mlp::PositionState& sample, double timestamp) {
                           return sample.ts < timestamp;
                         });
    std::size_t right = static_cast<std::size_t>(at_or_after - samples.begin());
    std::size_t left = 0;
    if (right == 0) {
      right = 1;
    } else if (right == samples.size()) {
      right = samples.size() - 1;
      left = right - 1;
    } else if (std::abs(samples[right].ts - timestamp_ms) < 1e-6 && right + 1 < samples.size()) {
      left = right - 1;
      ++right;
    } else {
      left = right - 1;
    }

    const double dt_ms = samples[right].ts - samples[left].ts;
    if (!std::isfinite(dt_ms) || dt_ms <= 0.0) return false;
    const std::array<double, 3> delta_cm = {
        samples[right].pos.x - samples[left].pos.x,
        samples[right].pos.y - samples[left].pos.y,
        samples[right].pos.z - samples[left].pos.z,
    };
    for (int axis = 0; axis < 3; ++axis) {
      // Position samples are centimetres and timestamps are milliseconds.
      velocity_m_s[axis] = 10.0 * delta_cm[axis] / dt_ms;
      if (!std::isfinite(velocity_m_s[axis])) return false;
    }
    return true;
  }

  void RefreshTarget() {
    last_target = {};
    predicted_trajectory.clear();
    if (stats.accepted < 21 || !track->isValid()) return;

    stats.last_error.clear();
    try {
      const float target_cm = static_cast<float>(100.0 * config.fixed_height_m);
      const float tolerance_cm = static_cast<float>(100.0 * config.height_tolerance_m);
      track->generatePredictedRpts(target_cm - tolerance_cm, target_cm + tolerance_cm,
                                   static_cast<float>(config.prediction_stride_ms),
                                   config.max_prediction_samples);
      const auto& hit = track->chooseHitPointByHeight(target_cm, tolerance_cm);
      const double latest_observation_ms = rows[next_row - 1].timestamp_ms;
      std::array<double, 3> velocity_m_s{};
      if (std::isfinite(hit.ts) && hit.ts > latest_observation_ms &&
          std::isfinite(hit.pos.x) && std::isfinite(hit.pos.y) &&
          std::isfinite(hit.pos.z) &&
          std::abs(hit.pos.z - target_cm) <= tolerance_cm + 1e-3f &&
          PredictedVelocityMps(hit.ts, velocity_m_s) && velocity_m_s[2] < 0.0) {
        last_target.valid = true;
        last_target.position_m = {hit.pos.x / 100.0, hit.pos.y / 100.0, hit.pos.z / 100.0};
        // Derive velocity from the same generated position samples. This keeps
        // SI output stable across ball-mlp's analytic and Taylor3 backends.
        last_target.incoming_velocity_m_s = velocity_m_s;
        last_target.hit_time_s = (hit.ts - first_timestamp_ms) / 1000.0;
        last_valid_hit_time_s = last_target.hit_time_s;
      }
    } catch (const std::exception& error) {
      stats.last_error = error.what();
      last_target = {};
    }

    try {
      // Generate the display curve after selecting the MPC target: this replaces
      // the track's narrow height-window cache without changing the saved target.
      track->generatePredictedRpts(0.0f, 0.0f,
                                   static_cast<float>(config.prediction_stride_ms),
                                   config.max_prediction_samples);
      const auto& samples = track->getRpts();
      predicted_trajectory.reserve(samples.size());
      for (const auto& sample : samples) {
        if (!std::isfinite(sample.ts) || !std::isfinite(sample.pos.x) ||
            !std::isfinite(sample.pos.y) || !std::isfinite(sample.pos.z))
          continue;
        predicted_trajectory.push_back({
            (sample.ts - first_timestamp_ms) / 1000.0,
            sample.pos.x / 100.0,
            sample.pos.y / 100.0,
            sample.pos.z / 100.0,
        });
      }
    } catch (const std::exception& error) {
      stats.last_error = error.what();
      predicted_trajectory.clear();
    }
  }

  bool UpdateUntil(double relative_time_s) {
    const auto begin = std::chrono::steady_clock::now();
    const auto record_duration = [&] {
      last_target.prediction_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    try {
      RequireFinite(relative_time_s, "relative_time_s");
      if (relative_time_s < 0.0) {
        record_duration();
        return last_target.valid;
      }
      const double absolute_limit_ms = first_timestamp_ms + 1000.0 * relative_time_s + 1e-6;
      bool changed = false;
      while (next_row < rows.size() && rows[next_row].timestamp_ms <= absolute_limit_ms) {
        const auto& row = rows[next_row];
        ball::mlp::SingleBall ball;
        ball.frame_id = static_cast<std::uint32_t>(next_row);
        ball.ts = row.timestamp_ms;
        ball.pos = {static_cast<float>(row.position_cm[0]), static_cast<float>(row.position_cm[1]),
                    static_cast<float>(row.position_cm[2])};
        ++stats.received;
        // Preassociated replay may explicitly bypass candidate-association gates;
        // addNode still rebuilds the fit and refreshes validity for every sample.
        if (track->addNode(ball, EstimatedHitTimeMs(row), config.preassociated_replay)) {
          ++stats.accepted;
          changed = true;
        } else {
          ++stats.rejected;
          stats.last_error = track->getLastError();
        }
        ++next_row;
      }
      if (changed) {
        RefreshTarget();
      } else if (!track->isValid()) {
        predicted_trajectory.clear();
      }
      record_duration();
      return last_target.valid;
    } catch (...) {
      record_duration();
      throw;
    }
  }

  bool Reference(double relative_time_s, double position_m[3], double velocity_m_s[3]) const {
    if (!position_m || !velocity_m_s || !std::isfinite(relative_time_s) || relative_time_s < 0.0 ||
        relative_time_s > end_time_s())
      return false;
    const double timestamp_ms = first_timestamp_ms + 1000.0 * relative_time_s;
    auto upper = std::upper_bound(rows.begin(), rows.end(), timestamp_ms,
                                  [](double value, const SourceRow& row) {
                                    return value < row.timestamp_ms;
                                  });
    const std::size_t right = std::clamp<std::size_t>(upper - rows.begin(), 1, rows.size() - 1);
    const auto& a = rows[right - 1];
    const auto& b = rows[right];
    const double dt_ms = b.timestamp_ms - a.timestamp_ms;
    const double alpha = std::clamp((timestamp_ms - a.timestamp_ms) / dt_ms, 0.0, 1.0);
    for (int axis = 0; axis < 3; ++axis) {
      const double position_cm =
          a.position_cm[axis] + alpha * (b.position_cm[axis] - a.position_cm[axis]);
      position_m[axis] = position_cm / 100.0;
      velocity_m_s[axis] = 10.0 * (b.position_cm[axis] - a.position_cm[axis]) / dt_ms;
    }
    return true;
  }

  double end_time_s() const { return (rows.back().timestamp_ms - first_timestamp_ms) / 1000.0; }

  Config config;
  std::vector<SourceRow> rows;
  double first_timestamp_ms = 0.0;
  std::size_t next_row = 0;
  std::unique_ptr<ball::mlp::BallMLP> predictor;
  std::unique_ptr<ball::mlp::SingleTrack> track;
  Target last_target;
  std::vector<std::array<double, 4>> predicted_trajectory;
  std::optional<double> last_valid_hit_time_s;
  Stats stats;
};

MeasuredShuttlePredictor::MeasuredShuttlePredictor(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

MeasuredShuttlePredictor::~MeasuredShuttlePredictor() = default;
MeasuredShuttlePredictor::MeasuredShuttlePredictor(MeasuredShuttlePredictor&&) noexcept = default;
MeasuredShuttlePredictor& MeasuredShuttlePredictor::operator=(MeasuredShuttlePredictor&&) noexcept =
    default;

bool MeasuredShuttlePredictor::UpdateUntil(double relative_time_s) {
  return impl_->UpdateUntil(relative_time_s);
}

const MeasuredShuttlePredictor::Target& MeasuredShuttlePredictor::target() const {
  return impl_->last_target;
}

const std::vector<std::array<double, 4>>& MeasuredShuttlePredictor::predicted_trajectory() const {
  return impl_->predicted_trajectory;
}

const MeasuredShuttlePredictor::Stats& MeasuredShuttlePredictor::stats() const {
  return impl_->stats;
}

double MeasuredShuttlePredictor::end_time_s() const { return impl_->end_time_s(); }

bool MeasuredShuttlePredictor::Reference(double relative_time_s, double position_m[3],
                                         double velocity_m_s[3]) const {
  return impl_->Reference(relative_time_s, position_m, velocity_m_s);
}

}  // namespace sckd
