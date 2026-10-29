#pragma once
namespace sckd::runtime {
// 固定仿真时间契约；不把名义tick频率声明为墙钟实时保证。
struct SynchronousSchedule {
  static constexpr double kTickSeconds=.001;
  static constexpr int kMaxTicks=6000;
  static double Time(int tick){return tick*kTickSeconds;}
  // 0: synchronous solve on every updated state, with no additional period wait.
  // A positive period retains fixed simulation-tick scheduling.
  static bool Replan(int tick, int period_ticks=0){return period_ticks==0 || tick%period_ticks==0;}
  static bool StateTrace(int tick){return tick%5==0;}
  static bool Viewer(int tick){return tick%17==0;}
};
}
