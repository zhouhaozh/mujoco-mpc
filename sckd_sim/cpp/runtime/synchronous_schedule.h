#pragma once
namespace sckd::runtime {
// 固定仿真时间契约；不把名义tick频率声明为墙钟实时保证。
struct SynchronousSchedule {
  static constexpr double kTickSeconds=.001;
  static constexpr int kMaxTicks=6000;
  static double Time(int tick){return tick*kTickSeconds;}
  static bool Replan(int tick){return tick%20==0;}
  static bool StateTrace(int tick){return tick%5==0;}
  static bool Viewer(int tick){return tick%17==0;}
};
}
