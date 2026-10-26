#pragma once
// 逐轴校对的纯逻辑状态机：不连接设备、不使能、不选择增益、不自动归零。
// 阈值只用于校对工具的异常检测，不是经过设备认证的机械/人身安全边界。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace sckd::io::axischeck {

constexpr double kDeg = 3.14159265358979323846 / 180.0;
using Joints = std::array<double, 5>;

struct Sample {
  Joints q{}, dq{}, stamp_s{};  // 每轴同一单调钟；stamp<=0表示未知。
  std::array<unsigned, 5> fault_bits{};
  std::array<int, 5> mode_state{};
};

struct Policy {
  // 方向测试允许保持静差/小幅联动；总行程和有效反馈检查仍生效。
  bool direction_only = false;
  double step_rad = kDeg;
  double max_excursion_rad = 3 * kDeg;
  // 反馈行程可独立于命令累计行程配置；0兼容旧规则，跟随max_excursion_rad。
  double max_feedback_excursion_rad = 0;
  double freshness_s = .10;
  double max_tracking_rad = 8 * kDeg;
  // SDK回home尚未到位时使用独立阈值，不套用点动跟踪误差。
  double startup_max_tracking_rad = 360 * kDeg;
  double max_other_motion_rad = 3 * kDeg;
  // 用户指定的反馈速度终止阈值；不是目标轨迹速度或SDK Start限速。
  double max_feedback_speed_rad_s = 1080 * kDeg;
  double max_tick_interval_s = .05;
  // 权威SDK配置：arm_stab_win_s=1, arm_stab_deg=1.5。
  double settle_time_s = 1.;
  double settle_span_rad = 1.5 * kDeg;
  double feedback_excursion_rad() const {
    return max_feedback_excursion_rad>0 ? max_feedback_excursion_rad : max_excursion_rad;
  }
};

struct Output {
  Joints q{}, dq{};
  bool initialized = false, moving = false, fault = false;
  bool settled = false;  // 五轴位置稳定窗通过；不代表精确到达命令角度。
  int active_joint = -1;  // 0..4，空闲时-1；界面显示时转换成J1..J5。
  std::string reason = "not_initialized";
  // 仅在一次动作获得稳定反馈后更新。稳定不等于执行了动作，调用方须
  // 检查response_direction后再决定是否继续自动循环或记录方向通过。
  uint64_t completed_move_id = 0;
  int completed_joint = -1;
  double completed_command_delta = 0, response_threshold_rad = 0;
  Joints feedback_start{}, feedback_end{}, feedback_delta{};
  std::string response_direction = "not_tested";
};

// 逐轴使用电机反馈时间戳；重复快照不会扩展窗口。
class PositionWindow {
 public:
  void Reset() { for(auto& q:history_) q.clear(); }
  bool Update(const Sample& s, const Policy& p) {
    bool ready=true;
    for(int j=0;j<5;++j) {
      auto& h=history_[j];
      if(h.empty() || s.stamp_s[j]>h.back().first) h.emplace_back(s.stamp_s[j],s.q[j]);
      // 保留窗口左边界之前的最近一点，避免不足1秒的窗被判通过。
      const double cutoff=s.stamp_s[j]-p.settle_time_s;
      while(h.size()>1 && h[1].first<=cutoff) h.pop_front();
      if(h.size()<2 || h.back().first-h.front().first<p.settle_time_s) ready=false;
      double low=h.front().second, high=low;
      for(const auto& point:h) { low=std::min(low,point.second); high=std::max(high,point.second); }
      if(high-low>=p.settle_span_rad) ready=false;
    }
    return ready;
  }
  Joints Mean() const {
    Joints mean{};
    for(int j=0;j<5;++j) { for(const auto& point:history_[j]) mean[j]+=point.second; if(!history_[j].empty()) mean[j]/=history_[j].size(); }
    return mean;
  }
 private:
  std::array<std::deque<std::pair<double,double>>,5> history_;
};

class Controller {
 public:
  explicit Controller(Policy policy = {}) : policy_(policy) {
    if(!std::isfinite(policy_.max_feedback_excursion_rad) || policy_.max_feedback_excursion_rad<0)
      throw std::invalid_argument("feedback excursion requires a finite nonnegative limit");
    for (double value : {policy_.step_rad,
         policy_.max_excursion_rad, policy_.freshness_s, policy_.max_tracking_rad,
         policy_.startup_max_tracking_rad,
         policy_.max_other_motion_rad, policy_.max_feedback_speed_rad_s,
         policy_.max_tick_interval_s, policy_.settle_time_s,
         policy_.settle_span_rad})
      if (!std::isfinite(value) || value <= 0)
        throw std::invalid_argument("axis-check policy requires finite positive limits");
    if (policy_.step_rad > policy_.max_excursion_rad)
      throw std::invalid_argument("axis-check step exceeds total excursion");

  }

  // 纯逻辑便捷入口。设备调用方必须用下一重载传入SDK最近已下发的目标，
  // 不能用带重力下垂的反馈反复重设保持位置。单个快照不能证明稳定。
  bool Initialize(const Sample& sample, double now) {
    return Initialize(sample, now, sample.q);
  }

  // 不允许通过重复初始化清故障或移动累计位移原点。成功后仍须连续Tick
  // 建立稳定窗口；initial_target必须是调用方核实的已下发位置，非新home。
  bool Initialize(const Sample& sample, double now, const Joints& initial_target) {
    return Initialize(sample,now,initial_target,sample.q);
  }
  bool Initialize(const Sample& sample, double now, const Joints& initial_target, const Joints& feedback_anchor) {
    if (output_.fault) return false;
    if (output_.initialized) return Reject("already_initialized");
    if (!CheckSample(sample, now, false)) return false;
    for (int j = 0; j < 5; ++j) {
      if (!std::isfinite(initial_target[j]))
        return Fault(AtJoint("nonfinite_initial_target", j));
      if (!std::isfinite(feedback_anchor[j]) || std::abs(feedback_anchor[j]-initial_target[j])>policy_.startup_max_tracking_rad)
        return Fault(AtJoint("invalid_feedback_anchor", j));
      // 只记录本轮固定静态偏差，绝不把下垂反馈写成新的SDK保持目标。
      feedback_offset_[j]=feedback_anchor[j]-initial_target[j];
      if (!CheckTarget(sample.q[j]-feedback_offset_[j], initial_target[j], initial_target[j],
                       initial_target[j], false, j)) return false;
    }
    origin_ = target_ = output_.q = initial_target;
    output_.dq.fill(0);
    output_.initialized = true;
    last_output_at_ = now;
    UpdateSettled(sample);
    output_.reason = "settling";
    return true;
  }

  bool StartMove(int joint, double signed_step, const Sample& sample, double now) {
    if (output_.fault) return false;
    if (!output_.initialized) return Reject("not_initialized");
    if (!CheckSample(sample, now, true)) return false;
    if (output_.moving) return Reject("busy");
    if (joint < 0 || joint >= 5) return Reject("invalid_joint");
    if (!std::isfinite(signed_step) ||
        std::abs(std::abs(signed_step) - policy_.step_rad) > StepTolerance())
      return Reject("invalid_step");
    UpdateSettled(sample);
    if (!output_.settled) return Reject("not_settled");
    // 只接受一个正/负配置步长，消除输入浮点容差造成的累计漂移。
    const double delta = signed_step > 0 ? policy_.step_rad : -policy_.step_rad;
    const double goal = target_[joint] + delta;
    if (!std::isfinite(goal) ||
        std::abs(goal - origin_[joint]) > policy_.max_excursion_rad + StepTolerance())
      return Reject("excursion_limit");
    move_start_ = target_;
    move_feedback_start_ = stability_.Mean();
    move_signed_step_ = delta;
    target_[joint] = goal;
    started_at_ = now;
    output_.q = move_start_;
    output_.dq.fill(0);
    output_.active_joint = joint;
    output_.moving = true;
    ResetSettled();
    last_output_at_ = now;
    output_.reason = "moving";
    return true;
  }

  Output Tick(const Sample& sample, double now) {
    if (output_.fault) return output_;
    if (!output_.initialized) { Reject("not_initialized"); return output_; }
    // 先校验整臂反馈，再推进目标。违规时保留最后已输出目标，不使用坏反馈。
    if (!CheckSample(sample, now, true)) return output_;
    if (output_.moving) {
      // 直接下发整步位置；vel=0为终点速度目标，不是电机运动限速。
      output_.q = target_;
      output_.dq.fill(0);
      UpdateSettled(sample);
      if (output_.settled) {
        CompleteObservation();
        output_.moving = false;
        output_.active_joint = -1;
      } else if (now-started_at_>15) {
        Fault("move_settle_timeout");
        return output_;
      }
    } else UpdateSettled(sample);
    last_output_at_ = now;
    output_.reason = output_.moving ? "moving" :
                     output_.settled ? "holding" : "settling";
    return output_;
  }

  // 软件停止不是失能/急停：锁存最后已输出的轨迹位置，速度目标立即为0。
  // 不向now外推、不拿反馈重设目标，避免延迟/陈旧反馈导致额外位置跳变。
  bool Stop(double now) {
    if (output_.fault) return false;
    if (!output_.initialized) return Reject("not_initialized");
    if (!CheckTime(now)) return false;
    last_now_ = now;
    target_ = output_.q;
    output_.dq.fill(0);
    output_.moving = false;
    output_.active_joint = -1;
    ResetSettled();
    last_output_at_ = now;
    output_.reason = "stopped";
    return true;
  }

  const Joints& origin() const { return origin_; }
  const Joints& feedback_offset() const { return feedback_offset_; }
  const Joints& target() const { return target_; }  // 当前段最终保持目标。
  // q是最后返回给调用方的目标；调用方须成功提交后再推进，不能生成后丢弃。
  const Output& status() const { return output_; }
  const Policy& policy() const { return policy_; }

 private:
  Policy policy_;
  Output output_;
  Joints origin_{}, target_{}, move_start_{}, last_stamps_{}, feedback_offset_{};
  Joints move_feedback_start_{};
  PositionWindow stability_;
  double started_at_ = 0, last_now_ = 0, last_output_at_ = 0;
  double move_signed_step_ = 0;
  bool have_time_ = false;

  double StepTolerance() const { return policy_.step_rad * 1e-9; }
  bool Reject(const std::string& reason) { output_.reason = reason; return false; }
  bool Fault(const std::string& reason) {
    output_.fault = true;
    output_.moving = false;
    output_.active_joint = -1;
    ResetSettled();
    output_.dq.fill(0);
    target_ = output_.q;
    return Reject(reason);
  }
  static std::string AtJoint(const char* reason, int j) {
    return std::string(reason) + ":J" + std::to_string(j + 1);
  }
  bool CheckTime(double now) {
    if (!std::isfinite(now) || now < 0) return Fault("invalid_time");
    if (have_time_ && now < last_now_) return Fault("time_reversed");
    // 以最后输出为锚；busy/无效输入的反馈检查不能掩盖停止Tick后的大跳步。
    if (output_.initialized && now - last_output_at_ > policy_.max_tick_interval_s)
      return Fault("control_time_gap");
    return true;
  }
  void ResetSettled() { output_.settled = false; stability_.Reset(); }
  void UpdateSettled(const Sample& sample) {
    output_.settled=stability_.Update(sample,policy_);
    if(output_.settled && !policy_.direction_only) {
      // 避免整步命令完全未执行却被静止窗误判完成；固定偏差已扣除。
      for(int j=0;j<5;++j) {
        const double limit=policy_.max_other_motion_rad;
        if(std::abs(sample.q[j]-feedback_offset_[j]-target_[j])>limit) output_.settled=false;
      }
    }
  }
  void CompleteObservation() {
    output_.feedback_start=move_feedback_start_;
    output_.feedback_end=stability_.Mean();
    for(int j=0;j<5;++j)
      output_.feedback_delta[j]=output_.feedback_end[j]-output_.feedback_start[j];
    output_.completed_joint=output_.active_joint;
    output_.completed_command_delta=move_signed_step_;
    output_.response_threshold_rad=std::min(.2*kDeg,.2*policy_.step_rad);
    const double actual=output_.feedback_delta[output_.completed_joint];
    output_.response_direction=std::abs(actual)<=output_.response_threshold_rad ? "no_response" :
        actual*move_signed_step_>0 ? "same" : "opposite";
    ++output_.completed_move_id;
  }
  bool CheckTarget(double q, double emitted, double target, double origin,
                   bool selected, int j) {
    // 与已输出位置比较，不能用尚未发送的未来轨迹点掩盖跟踪误差。
    // 直接位置命令期间，整步起点到终点是预期运动区间；不能将10°阶跃
    // 当成8°跟踪故障。区间外仍执行原8°误差和总行程保护。
    const double reference = selected && output_.moving
        ? std::clamp(q, std::min(move_start_[j],target), std::max(move_start_[j],target))
        : emitted;
    if (!policy_.direction_only && std::abs(q - reference) > policy_.max_tracking_rad)
      return Fault(AtJoint("tracking_error", j));
    if (!policy_.direction_only && !selected && std::abs(q - target) > policy_.max_other_motion_rad)
      return Fault(AtJoint("other_axis_motion", j));
    if (std::abs(q - origin) > policy_.feedback_excursion_rad() + StepTolerance())
      return Fault(AtJoint("feedback_excursion", j));
    return true;
  }
  bool CheckSample(const Sample& sample, double now, bool check_targets) {
    if (!CheckTime(now)) return false;
    for (int j = 0; j < 5; ++j) {
      if (!std::isfinite(sample.q[j]) || !std::isfinite(sample.dq[j]))
        return Fault(AtJoint("nonfinite_feedback", j));
      const double stamp = sample.stamp_s[j];
      if (!std::isfinite(stamp) || stamp <= 0)
        return Fault(AtJoint("unknown_stamp", j));
      if (stamp > now) return Fault(AtJoint("future_feedback", j));
      if (now - stamp > policy_.freshness_s)
        return Fault(AtJoint("stale_feedback", j));
      if (have_time_ && stamp < last_stamps_[j])
        return Fault(AtJoint("feedback_time_reversed", j));
      if (sample.fault_bits[j] != 0) return Fault(AtJoint("joint_fault", j));
      if (sample.mode_state[j] != 2) return Fault(AtJoint("mode_not_ready", j));
      if (std::abs(sample.dq[j]) > policy_.max_feedback_speed_rad_s)
        return Fault(AtJoint("feedback_speed", j));
      if (check_targets) {
        if (!CheckTarget(sample.q[j]-feedback_offset_[j], output_.q[j], target_[j], origin_[j],
                         j == output_.active_joint, j)) return false;
      }
    }
    last_now_ = now;
    last_stamps_ = sample.stamp_s;
    have_time_ = true;
    return true;
  }
};

}  // namespace sckd::io::axischeck
