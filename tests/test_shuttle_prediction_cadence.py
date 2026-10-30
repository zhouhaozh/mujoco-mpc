import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_SOURCE = ROOT / "sckd_sim/cpp/estimation/measured_shuttle_predictor.cpp"
CPP_INCLUDE = ROOT / "sckd_sim/cpp"


BALL_MLP_STUB = r"""
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ball::mlp {

enum class TrackTimeSource { LatestNodeTime };

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct SingleBall {
  std::uint32_t frame_id = 0;
  double ts = 0.0;
  Vec3 pos;
};

struct PositionState {
  double ts = 0.0;
  Vec3 pos;
};

struct FramesFillerParams {
  struct CommonParams {
    TrackTimeSource track_time_source = TrackTimeSource::LatestNodeTime;
    bool check_timeout = false;
  } common_params;

  static FramesFillerParams createFps1000FillerParams() { return {}; }
};

class SingleTrack {
 public:
  bool addNode(const SingleBall& ball, float, bool) {
    nodes_.push_back(ball);
    return true;
  }

  bool isValid() const {
    return nodes_.size() >= 21 && nodes_.back().pos.x >= 0.0f;
  }

  const std::string& getLastError() const { return error_; }

  void generatePredictedRpts(float min_height, float max_height, float, std::size_t) {
    predictions_.clear();
    if (!isValid()) return;
    const float target_height = min_height > 0.0f ? (min_height + max_height) / 2.0f : 80.0f;
    const double base = nodes_.back().ts + 10.0;
    predictions_.push_back({base, {nodes_.back().pos.x, nodes_.back().pos.y, target_height}});
    predictions_.push_back({base + 1.0,
                            {nodes_.back().pos.x + 1.0f, nodes_.back().pos.y,
                             target_height - 1.0f}});
  }

  const PositionState& chooseHitPointByHeight(float, float) const {
    return predictions_.front();
  }

  const std::vector<PositionState>& getRpts() const { return predictions_; }

 private:
  std::vector<SingleBall> nodes_;
  std::vector<PositionState> predictions_;
  std::string error_;
};

class BallMLP {
 public:
  explicit BallMLP(const FramesFillerParams&) {}
  void loadNet20Model(const std::string&, const std::string&) {}
  void setTrackTimeSource(TrackTimeSource) {}
  std::unique_ptr<SingleTrack> createSingleTrack() {
    return std::make_unique<SingleTrack>();
  }
};

}  // namespace ball::mlp
"""


HARNESS = r"""
#include "estimation/measured_shuttle_predictor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

bool Near(double actual, double expected) {
  return std::abs(actual - expected) < 1e-9;
}

}  // namespace

int main(int argc, char** argv) {
  Require(argc == 4, "expected trajectory, model, and normalizer paths");
  sckd::MeasuredShuttlePredictor::Config config;
  config.trajectory_path = argv[1];
  config.net20_model_path = argv[2];
  config.net20_normalizer_path = argv[3];
  config.fixed_height_m = 0.8;
  config.height_tolerance_m = 0.02;
  config.preassociated_replay = true;
  sckd::MeasuredShuttlePredictor predictor(config);

  predictor.UpdateUntil(-0.001);
  Require(predictor.stats().received == 0, "default refresh read observations before time zero");
  Require(!predictor.target().valid && predictor.predicted_trajectory().empty(),
          "first refresh without observations published a prediction");

  for (int millisecond = 0; millisecond < 20; ++millisecond) {
    predictor.UpdateUntil(millisecond / 1000.0, false);
  }
  Require(predictor.stats().received == 20, "1ms ingestion lost observations before refresh");
  Require(!predictor.target().valid, "ingestion-only call unexpectedly refreshed target");
  Require(predictor.predicted_trajectory().empty(),
          "ingestion-only call unexpectedly refreshed trajectory");

  predictor.UpdateUntil(0.020, true);
  Require(predictor.stats().received == 21, "refresh did not include the boundary observation");
  Require(predictor.target().valid, "first prediction refresh did not publish a target");
  Require(Near(predictor.target().hit_time_s, 0.030),
          "prediction used observations beyond the requested simulation time");
  const auto held_target = predictor.target();
  const auto held_trajectory = predictor.predicted_trajectory();

  for (int millisecond = 21; millisecond < 24; ++millisecond) {
    predictor.UpdateUntil(millisecond / 1000.0, false);
  }
  Require(predictor.stats().received == 24, "1ms ingestion did not continue between refreshes");
  Require(predictor.target().valid == held_target.valid &&
              predictor.target().position_m == held_target.position_m &&
              predictor.target().incoming_velocity_m_s == held_target.incoming_velocity_m_s &&
              Near(predictor.target().hit_time_s, held_target.hit_time_s) &&
              predictor.target().prediction_ms == held_target.prediction_ms,
          "target changed inside the 4ms hold interval");
  Require(predictor.predicted_trajectory() == held_trajectory,
          "trajectory changed inside the 4ms hold interval");

  predictor.UpdateUntil(0.024, true);
  Require(predictor.stats().received == 24,
          "refresh without a new row changed the ingested observation count");
  Require(Near(predictor.target().hit_time_s, 0.033),
          "refresh without a new row dropped accumulated observations");
  const auto valid_target = predictor.target();
  const auto valid_trajectory = predictor.predicted_trajectory();

  predictor.UpdateUntil(0.025, false);
  Require(predictor.stats().received == 25, "invalid observation was not ingested");
  Require(predictor.target().valid == valid_target.valid &&
              predictor.target().position_m == valid_target.position_m &&
              predictor.target().incoming_velocity_m_s == valid_target.incoming_velocity_m_s &&
              Near(predictor.target().hit_time_s, valid_target.hit_time_s) &&
              predictor.target().prediction_ms == valid_target.prediction_ms,
          "invalid ingestion cleared target before the refresh boundary");
  Require(predictor.predicted_trajectory() == valid_trajectory,
          "invalid ingestion cleared trajectory before the refresh boundary");

  predictor.UpdateUntil(0.028);
  Require(predictor.stats().received == 28, "invalid refresh did not ingest through its boundary");
  Require(!predictor.target().valid, "invalid track retained a stale target after refresh");
  Require(predictor.predicted_trajectory().empty(),
          "invalid track retained a stale trajectory after refresh");
}
"""


class ShuttlePredictionCadenceTests(unittest.TestCase):
    def test_ingestion_runs_each_millisecond_while_prediction_refreshes_every_four(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is unavailable")

        with tempfile.TemporaryDirectory(prefix="shuttle-cadence-") as directory:
            temporary = Path(directory)
            stub = temporary / "ball-mlp/ball_mlp.h"
            stub.parent.mkdir()
            stub.write_text(textwrap.dedent(BALL_MLP_STUB))
            harness = temporary / "cadence_test.cpp"
            harness.write_text(textwrap.dedent(HARNESS))
            trajectory = temporary / "trajectory.txt"
            trajectory.write_text(
                "".join(
                    f"{1000 + index} {index if index < 25 else -1} 0 {100 - index}\n"
                    for index in range(30) if index != 24
                )
            )
            model = temporary / "model.engine"
            normalizer = temporary / "normalizer.txt"
            model.touch()
            normalizer.touch()
            binary = temporary / "cadence_test"

            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(temporary),
                    "-I",
                    str(CPP_INCLUDE),
                    str(PREDICTOR_SOURCE),
                    str(harness),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)

            result = subprocess.run(
                [str(binary), str(trajectory), str(model), str(normalizer)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
