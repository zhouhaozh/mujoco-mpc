#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {

void UpdateInputs(InputContext c, double ts) {
  auto& be = c.be;
  auto& predictor = c.predictor;
  auto& measured = c.measured;
  const auto& deduplicate_obs = c.deduplicate_obs;
  auto& snapshot=c.input_snapshot;
  auto& last_obs_t = c.last_obs_t;
  auto& q_cur = c.q_cur;
  auto& dq_cur = c.dq_cur;
  // 观测 → 预测;当前臂构型取实测
  BallObs obs;
  auto consume=[&](const BallObs& value){
    StageProfile::Measure("observation_update", [&]{predictor.Update(value);});
    auto identity=snapshot.ball.identity;
    if(!snapshot.have_ball||value.t!=identity.captured.seconds){
      ++identity.observation_seq;
      identity.received={ts,"simulation"};
    }
    snapshot.ball=BallObservation::FromLegacy(value,std::move(identity));
    snapshot.have_ball=true;
    last_obs_t=value.t;
  };
  if(measured) {
    while(measured->TakeReceived(ts,&obs)) {
      consume(obs);
    }
  } else if (be.get_ball_obs(&obs) && (!deduplicate_obs || obs.t > last_obs_t)) {
    consume(obs);
  }
  snapshot.robot.sampled={ts,"simulation"};
  snapshot.robot.state_seq=static_cast<Sequence>(std::llround(ts/.001));
  snapshot.robot.base=be.base_state();
  snapshot.robot.valid=true;
  snapshot.robot.base_source=RobotStateSource::kSimulationTruth;
  snapshot.robot.joint_position_source.fill(RobotStateSource::kSimulationTruth);
  snapshot.robot.joint_velocity_source.fill(RobotStateSource::kSimulationTruth);
  for (int j = 0; j < 5; ++j) {
    const JointReport jr = be.joint_measured(AxisArm(j));
    snapshot.robot.joints[AxisArm(j)]=jr;
    snapshot.robot.joint_position_source[AxisArm(j)]=be.motor_config().truth_feedback
        ? RobotStateSource::kSimulationTruth : RobotStateSource::kMeasured;
    q_cur[j] = jr.q; dq_cur[j] = jr.dq;
  }
  // 新增状态快照不能多消费带噪传感器样本，否则会推进SimBackend共享RNG，
  // 改变后续球观测。底盘轴仅记录显式真值，保留原控制的观测调用次数。
  for(int axis=0;axis<12;++axis)snapshot.robot.joints[axis]=be.joint(axis);


}
}
