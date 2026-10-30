#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sckd {

// Streams recorded shuttle observations into ball-mlp without exposing future
// samples. Source positions are centimetres and timestamps are milliseconds;
// public results use SI units and a clock relative to the first source sample.
class MeasuredShuttlePredictor {
 public:
  struct Config {
    std::string trajectory_path;
    std::string net20_model_path;
    std::string net20_normalizer_path;
    double fixed_height_m = 0.8;
    double height_tolerance_m = 0.05;
    double prediction_stride_ms = 1.0;
    std::size_t max_prediction_samples = 2000;
    double initial_estimated_hit_s = 0.5;
    bool preassociated_replay = false;
  };

  struct Target {
    bool valid = false;
    std::array<double, 3> position_m{};
    std::array<double, 3> incoming_velocity_m_s{};
    double hit_time_s = 0.0;
    double prediction_ms = 0.0;
  };

  struct Stats {
    std::size_t received = 0;
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::string input_policy = "normal_association";
    std::string last_error;
  };

  explicit MeasuredShuttlePredictor(Config config);
  ~MeasuredShuttlePredictor();

  MeasuredShuttlePredictor(const MeasuredShuttlePredictor&) = delete;
  MeasuredShuttlePredictor& operator=(const MeasuredShuttlePredictor&) = delete;
  MeasuredShuttlePredictor(MeasuredShuttlePredictor&&) noexcept;
  MeasuredShuttlePredictor& operator=(MeasuredShuttlePredictor&&) noexcept;

  // Adds only source rows whose relative timestamp is <= relative_time_s.
  // When false, retain the published target/trajectory until a refreshing call.
  // Returns true when a valid descending fixed-height target is available.
  bool UpdateUntil(double relative_time_s, bool refresh_prediction = true);

  const Target& target() const;
  // Rows are {relative_time_s, x_m, y_m, z_m} from the current model forecast.
  const std::vector<std::array<double, 4>>& predicted_trajectory() const;
  const Stats& stats() const;
  double end_time_s() const;

  // Display/reference helper. It may interpolate the complete recording, but
  // its output is never fed to ball-mlp. Returns false outside the recording.
  bool Reference(double relative_time_s, double position_m[3], double velocity_m_s[3]) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sckd
