#pragma once
#include "runtime/arm_axis_check.h"
namespace sckd::io::axischeck {
// 只审核SDK Start返回后的就位过程；启动位置误差使用独立阈值。
class StartupCheck {
 public:
  explicit StartupCheck(double start, Joints target, Policy policy = {})
      : start_(start), target_(target), policy_(policy) {}
  bool Update(const Sample& s, double now) {
    if (fault_) return false;
    auto Fail = [&](std::string reason) { fault_=true; reason_=reason; return false; };
    if (!std::isfinite(now) || now<start_ || (last_>=0 && (now<=last_ || now-last_>policy_.max_tick_interval_s)))
      return Fail("startup_control_gap");
    last_=now;

    for(int j=0;j<5;++j) {
      const auto joint=":J"+std::to_string(j+1);
      if (!std::isfinite(s.q[j]) || !std::isfinite(s.dq[j]) || !std::isfinite(s.stamp_s[j]) ||
          s.stamp_s[j]<=0 || s.stamp_s[j]>now || now-s.stamp_s[j]>policy_.freshness_s ||
          s.fault_bits[j] || s.mode_state[j]!=2) return Fail("startup_invalid_feedback"+joint);
      if(std::abs(s.dq[j])>policy_.max_feedback_speed_rad_s) return Fail("startup_feedback_speed"+joint);
      if(std::abs(s.q[j]-target_[j])>policy_.startup_max_tracking_rad) return Fail("startup_tracking_error"+joint);
      if(s.stamp_s[j]<previous_[j]) return Fail("startup_feedback_time_reversed"+joint);
    }
    if(now-start_>15) return Fail("startup_settle_timeout");
    ready_=stability_.Update(s,policy_);
    previous_=s.stamp_s;
    reason_=ready_ ? "startup_settled" : "startup_settling";
    return ready_;
  }
  Joints feedback_anchor() const { return stability_.Mean(); }
  bool fault() const { return fault_; }
  const std::string& reason() const { return reason_; }
 private:
  double start_, last_=-1;
  PositionWindow stability_;
  Joints target_, previous_{};
  Policy policy_;
  bool fault_=false, ready_=false;
  std::string reason_="startup_settling";
};
}
