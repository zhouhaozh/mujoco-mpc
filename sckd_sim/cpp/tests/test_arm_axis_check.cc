// Ubuntu执行；纯逻辑，不使能设备。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <limits>
#include "runtime/arm_axis_check.h"
using namespace sckd::io::axischeck;
namespace {
const Joints kOrigin{.12,-.25,.31,-.42,.53};
Sample Healthy(const Joints& q,double now) {
  Sample s; s.q=q; s.stamp_s.fill(now-.001); s.mode_state.fill(2); return s;
}
void Settle(Controller& c,double& now) {
  for(int i=0;i<110;++i) { now+=.01; REQUIRE_FALSE(c.Tick(Healthy(c.status().q,now),now).fault); }
  REQUIRE(c.status().settled); REQUIRE_FALSE(c.status().moving);
}
void Init(Controller& c,double& now) { REQUIRE(c.Initialize(Healthy(kOrigin,now),now)); Settle(c,now); }
void SettleAt(Controller& c,double& now,const Joints& feedback) {
  for(int i=0;i<110;++i) { now+=.01; REQUIRE_FALSE(c.Tick(Healthy(feedback,now),now).fault); }
  REQUIRE(c.status().settled); REQUIRE_FALSE(c.status().moving);
}
}
TEST_CASE("direction mode records measured increments while tolerating target and coupled-axis offsets") {
  Policy p; p.direction_only=true; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); double now=2;
  auto measured=kOrigin; measured[1]-=21.64*kDeg; measured[2]-=4.56*kDeg;
  REQUIRE(c.Initialize(Healthy(measured,now),now,kOrigin));
  SettleAt(c,now,measured); const auto offset=c.feedback_offset();
  REQUIRE(c.StartMove(1,p.step_rad,Healthy(measured,now),now));
  now+=.004; REQUIRE_FALSE(c.Tick(Healthy(measured,now),now).fault);
  measured[1]+=6*kDeg; measured[2]+=4.2*kDeg;
  SettleAt(c,now,measured);
  const auto out=c.status();
  CHECK(out.completed_move_id==1); CHECK(out.completed_joint==1);
  CHECK(out.completed_command_delta==doctest::Approx(10*kDeg));
  CHECK(out.feedback_delta[1]==doctest::Approx(6*kDeg));
  CHECK(out.feedback_delta[2]==doctest::Approx(4.2*kDeg));
  CHECK(out.response_direction=="same");
  CHECK(out.q[1]==doctest::Approx(kOrigin[1]+10*kDeg));
  CHECK(c.origin()==kOrigin); CHECK(c.feedback_offset()==offset);
  SettleAt(c,now,measured); CHECK(c.status().completed_move_id==1);
}
TEST_CASE("direction mode distinguishes no response from stability without terminating the session") {
  Policy p; p.direction_only=true; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); double now=2; Init(c,now);
  REQUIRE(c.StartMove(0,p.step_rad,Healthy(kOrigin,now),now));
  now+=.004; REQUIRE_FALSE(c.Tick(Healthy(kOrigin,now),now).fault);
  SettleAt(c,now,kOrigin);
  CHECK(c.status().response_direction=="no_response");
  CHECK(c.status().completed_move_id==1);
  for(double change:c.status().feedback_delta) CHECK(change==doctest::Approx(0).epsilon(1e-12));
  CHECK(c.status().q[0]==doctest::Approx(kOrigin[0]+10*kDeg));
  SettleAt(c,now,kOrigin); CHECK(c.status().completed_move_id==1);
  CHECK_FALSE(c.StartMove(0,p.step_rad,Healthy(kOrigin,now),now));
  CHECK(c.status().reason=="excursion_limit");
}
TEST_CASE("direction response uses command sign and a measurable displacement threshold") {
  for(double command_sign:{-1.,1.}) for(double measured_deg:{-.8,-.1,0.,.1,.8}) {
    Policy p; p.direction_only=true;
    Controller c(p); double now=2; Init(c,now);
    REQUIRE(c.StartMove(3,command_sign*p.step_rad,Healthy(kOrigin,now),now));
    now+=.004; REQUIRE_FALSE(c.Tick(Healthy(kOrigin,now),now).fault);
    auto feedback=kOrigin; feedback[3]+=measured_deg*kDeg;
    SettleAt(c,now,feedback);
    CHECK(c.status().response_threshold_rad==doctest::Approx(.2*kDeg));
    const std::string expected=std::abs(measured_deg)<=.2 ? "no_response" :
        measured_deg*command_sign>0 ? "same" : "opposite";
    CHECK(c.status().response_direction==expected);
  }
}
TEST_CASE("direction mode accepts stable J5 return residual but reports its actual response") {
  Policy p; p.direction_only=true; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); double now=2; auto measured=kOrigin; measured[4]-=2.788*kDeg;
  REQUIRE(c.Initialize(Healthy(measured,now),now,kOrigin)); SettleAt(c,now,measured);
  REQUIRE(c.StartMove(4,p.step_rad,Healthy(measured,now),now));
  now+=.004; REQUIRE_FALSE(c.Tick(Healthy(measured,now),now).fault);
  measured[4]+=8.67*kDeg; SettleAt(c,now,measured);
  CHECK(c.status().response_direction=="same");
  REQUIRE(c.StartMove(4,-p.step_rad,Healthy(measured,now),now));
  now+=.004; REQUIRE_FALSE(c.Tick(Healthy(measured,now),now).fault);
  measured[4]-=1.333*kDeg; SettleAt(c,now,measured);
  CHECK(c.status().completed_move_id==2); CHECK(c.status().response_direction=="same");
  // 回程小于1.5°稳定窗极差，首次完成均值可能仍含窗口左边界的旧点。
  // 记录必须与该次实际冻结的两个均值一致，而不是假定等于最终单点。
  CHECK(c.status().feedback_delta[4]==doctest::Approx(
      c.status().feedback_end[4]-c.status().feedback_start[4]));
  CHECK(std::abs(c.status().feedback_delta[4]/kDeg+1.333)<.02);
  CHECK(c.status().q[4]==doctest::Approx(kOrigin[4]));
  CHECK(measured[4]-c.feedback_offset()[4]-c.target()[4]==doctest::Approx(7.337*kDeg));
}
TEST_CASE("direction mode retains fixed feedback excursion and invalid feedback guards") {
  for(int mode=0;mode<9;++mode) {
    Policy p; p.direction_only=true; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
    Controller c(p); double now=2; Init(c,now);
    REQUIRE(c.StartMove(0,p.step_rad,Healthy(kOrigin,now),now));
    now+=.004; REQUIRE_FALSE(c.Tick(Healthy(kOrigin,now),now).fault);
    const auto target=c.status().q; now+=.004; auto f=Healthy(kOrigin,now);
    if(mode==0) f.q[2]+=12.01*kDeg;
    if(mode==1) f.stamp_s[2]=0;
    if(mode==2) f.stamp_s[2]=now-.101;
    if(mode==3) f.fault_bits[2]=1;
    if(mode==4) f.mode_state[2]=0;
    if(mode==5) f.dq[2]=1081*kDeg;
    if(mode==6) f.q[2]=std::numeric_limits<double>::quiet_NaN();
    if(mode==7) f.stamp_s[2]=now+.001;
    if(mode==8) now+=.2;
    const auto out=c.Tick(f,now); CHECK(out.fault); CHECK(out.q==target);
    CHECK(out.completed_move_id==0); CHECK(out.response_direction=="not_tested");
  }
}
TEST_CASE("direction feedback excursion can be 45 degrees while command excursion stays 12") {
  Policy p; p.direction_only=true; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  p.max_feedback_excursion_rad=45*kDeg;
  CHECK(p.feedback_excursion_rad()==doctest::Approx(45*kDeg));
  for(double displacement:{-45.,45.,-45.01,45.01}) {
    Controller c(p); double now=2; Init(c,now); auto feedback=kOrigin;
    feedback[2]+=displacement*kDeg; now+=.004;
    const auto out=c.Tick(Healthy(feedback,now),now);
    if(std::abs(displacement)<=45) CHECK_FALSE(out.fault);
    else { CHECK(out.fault); CHECK(out.reason=="feedback_excursion:J3"); }
  }
  Controller c(p); double now=2; Init(c,now);
  REQUIRE(c.StartMove(0,p.step_rad,Healthy(kOrigin,now),now));
  now+=.004; REQUIRE_FALSE(c.Tick(Healthy(kOrigin,now),now).fault); Settle(c,now);
  CHECK_FALSE(c.StartMove(0,p.step_rad,Healthy(c.target(),now),now));
  CHECK(c.status().reason=="excursion_limit");
  CHECK(c.status().q[0]==doctest::Approx(kOrigin[0]+10*kDeg));
}
TEST_CASE("unspecified feedback excursion follows legacy command excursion") {
  Policy p; CHECK(p.max_feedback_excursion_rad==0);
  p.max_excursion_rad=2*kDeg;
  CHECK(p.feedback_excursion_rad()==doctest::Approx(2*kDeg));
  Controller c(p); double now=2; Init(c,now); auto feedback=kOrigin;
  // 单轴2.01°仍小于原tracking/other轴阈值，因此精确命中旧反馈行程门。
  feedback[2]+=2.01*kDeg; now+=.004;
  CHECK(c.Tick(Healthy(feedback,now),now).reason=="feedback_excursion:J3");
  for(double invalid:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    p.max_feedback_excursion_rad=invalid;
    CHECK_THROWS_AS(Controller{p},std::invalid_argument);
  }
}
TEST_CASE("direction thirty degree steps return every axis and reject a cumulative sixty degree command") {
  Policy p; p.direction_only=true; p.step_rad=30*kDeg;
  p.max_excursion_rad=45*kDeg; p.max_feedback_excursion_rad=45*kDeg;
  for(int j=0;j<5;++j) {
    Controller c(p); double now=2; Init(c,now);
    REQUIRE(c.StartMove(j,p.step_rad,Healthy(kOrigin,now),now));
    now+=.004; const auto first=c.Tick(Healthy(kOrigin,now),now);
    REQUIRE_FALSE(first.fault); CHECK(first.moving);
    CHECK(first.q[j]==doctest::Approx(kOrigin[j]+30*kDeg));
    CHECK(first.dq==Joints{});
    for(int k=0;k<5;++k) if(k!=j) CHECK(first.q[k]==kOrigin[k]);
    Settle(c,now);
    CHECK(c.status().completed_move_id==1); CHECK(c.status().completed_joint==j);
    CHECK(c.status().response_direction=="same");
    CHECK(c.status().completed_command_delta==doctest::Approx(30*kDeg));
    CHECK(c.status().feedback_delta[j]==doctest::Approx(30*kDeg));
    const auto positive_target=c.status().q;
    CHECK_FALSE(c.StartMove(j,p.step_rad,Healthy(positive_target,now),now));
    CHECK(c.status().reason=="excursion_limit"); CHECK(c.status().q==positive_target);
    REQUIRE(c.StartMove(j,-p.step_rad,Healthy(positive_target,now),now));
    now+=.004; REQUIRE_FALSE(c.Tick(Healthy(positive_target,now),now).fault);
    Settle(c,now);
    CHECK(c.status().completed_move_id==2); CHECK(c.status().completed_joint==j);
    CHECK(c.status().response_direction=="same");
    CHECK(c.status().completed_command_delta==doctest::Approx(-30*kDeg));
    CHECK(c.status().feedback_delta[j]==doctest::Approx(-30*kDeg));
    for(int k=0;k<5;++k) CHECK(c.status().q[k]==doctest::Approx(kOrigin[k]));
    CHECK(c.origin()==kOrigin);
  }
}
TEST_CASE("independent command limit accepts exactly forty five degrees in either direction") {
  Policy p; p.direction_only=true; p.step_rad=15*kDeg;
  p.max_excursion_rad=45*kDeg; p.max_feedback_excursion_rad=45*kDeg;
  for(double sign:{-1.,1.}) {
    Controller c(p); double now=2; Init(c,now);
    for(int n=1;n<=3;++n) {
      const auto start=c.status().q;
      REQUIRE(c.StartMove(0,sign*p.step_rad,Healthy(start,now),now));
      now+=.004; REQUIRE_FALSE(c.Tick(Healthy(start,now),now).fault); Settle(c,now);
      CHECK(c.status().q[0]==doctest::Approx(kOrigin[0]+sign*n*15*kDeg));
      CHECK(c.status().completed_move_id==static_cast<uint64_t>(n));
      CHECK(c.status().response_direction=="same");
    }
    const auto boundary=c.status().q;
    CHECK_FALSE(c.StartMove(0,sign*p.step_rad,Healthy(boundary,now),now));
    CHECK(c.status().reason=="excursion_limit"); CHECK(c.status().q==boundary);
  }
}
TEST_CASE("direct ten degree targets on every axis in both directions") {
  Policy p; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  for(int j=0;j<5;++j) for(double sign:{-1.,1.}) {
    Controller c(p); double now=2; Init(c,now);
    REQUIRE(c.StartMove(j,sign*p.step_rad,Healthy(kOrigin,now),now));
    // First output is full position step, no interpolator or prescribed duration.
    now+=.004; auto out=c.Tick(Healthy(kOrigin,now),now);
    REQUIRE_FALSE(out.fault); REQUIRE(out.moving); CHECK_FALSE(out.settled);
    CHECK(out.q[j]==doctest::Approx(kOrigin[j]+sign*p.step_rad)); CHECK(out.dq==Joints{});
    for(int k=0;k<5;++k) if(k!=j) CHECK(out.q[k]==kOrigin[k]);
    Settle(c,now);
    CHECK_FALSE(c.StartMove(j,sign*p.step_rad,Healthy(c.target(),now),now));
    CHECK(c.status().reason=="excursion_limit");
    REQUIRE(c.StartMove(j,-sign*p.step_rad,Healthy(c.target(),now),now));
    now+=.004; out=c.Tick(Healthy(c.status().q,now),now); REQUIRE_FALSE(out.fault);
    for(int k=0;k<5;++k) CHECK(out.q[k]==doctest::Approx(kOrigin[k]));
    Settle(c,now);
  }
}
TEST_CASE("stationary feedback offset does not alter commands or drift anchors") {
  Policy p; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); double now=2;
  auto q=kOrigin; q[1]-=21.64*kDeg; q[2]-=4.56*kDeg; q[4]-=2.72*kDeg;
  REQUIRE(c.Initialize(Healthy(q,now),now,kOrigin)); const auto offset=c.feedback_offset();
  for(int n=0;n<110;++n) { now+=.01; auto f=Healthy(q,now); f.dq.fill(14*kDeg); CHECK_FALSE(c.Tick(f,now).fault); }
  CHECK(c.status().settled); CHECK(c.status().q==kOrigin);
  REQUIRE(c.StartMove(1,p.step_rad,Healthy(q,now),now));
  now+=.004; CHECK_FALSE(c.Tick(Healthy(q,now),now).fault);
  CHECK(c.status().q[1]==doctest::Approx(kOrigin[1]+p.step_rad));
  // 目标改变10°而实际只变6°，即使稳定，也不能先通过再在holding阶段故障。
  q[1]+=6*kDeg;
  for(int n=0;n<110;++n) { now+=.01; auto out=c.Tick(Healthy(q,now),now); REQUIRE_FALSE(out.fault); CHECK_FALSE(out.settled); }
  q[1]+=4*kDeg;
  for(int n=0;n<120;++n) { now+=.01; REQUIRE_FALSE(c.Tick(Healthy(q,now),now).fault); }
  CHECK(c.status().settled); REQUIRE(c.Stop(now)); CHECK(c.feedback_offset()==offset);
  CHECK_FALSE(c.Initialize(Healthy(q,now),now,kOrigin)); CHECK(c.feedback_offset()==offset);
}
TEST_CASE("direct move timeout and software stop retain last emitted target") {
  Policy p; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); double now=2; Init(c,now); REQUIRE(c.StartMove(0,p.step_rad,Healthy(kOrigin,now),now));
  SUBCASE("timeout") {
    for(int i=0;i<1502 && !c.status().fault;++i) { now+=.01; c.Tick(Healthy(kOrigin,now),now); }
    CHECK(c.status().fault); CHECK(c.status().reason=="move_settle_timeout");
  }
  SUBCASE("stop") {
    now+=.004; c.Tick(Healthy(kOrigin,now),now); auto sent=c.status().q;
    REQUIRE(c.Stop(now)); CHECK(c.status().q==sent); CHECK(c.status().dq==Joints{}); CHECK_FALSE(c.status().moving);
  }
}
TEST_CASE("invalid feedback freezes direct command and latches failure") {
  for(int mode=0;mode<11;++mode) {
    Controller c; double now=2; Init(c,now); REQUIRE(c.StartMove(0,kDeg,Healthy(kOrigin,now),now));
    now+=.004; c.Tick(Healthy(kOrigin,now),now); auto sent=c.status().q;
    now+=.004; auto f=Healthy(kOrigin,now);
    if(mode==0) f.q[4]=std::numeric_limits<double>::quiet_NaN();
    if(mode==1) f.dq[4]=std::numeric_limits<double>::infinity();
    if(mode==2) f.stamp_s[4]=0;
    if(mode==3) f.stamp_s[4]=now+.001;
    if(mode==4) f.stamp_s[4]=now-.101;
    if(mode==5) f.fault_bits[4]=1;
    if(mode==6) f.mode_state[4]=0;
    if(mode==7) f.dq[4]=1081*kDeg;
    if(mode==8) f.q[4]+=3.1*kDeg;
    if(mode==9) f.q[0]-=9*kDeg;
    if(mode==10) now+=.2;
    auto out=c.Tick(f,now); CHECK(out.fault); CHECK(out.q==sent); CHECK(out.dq==Joints{});
    CHECK_FALSE(c.Initialize(Healthy(kOrigin,now),now)); CHECK_FALSE(c.StartMove(1,kDeg,Healthy(kOrigin,now),now));
  }
}
TEST_CASE("policy and command validation remain active") {
  Policy p; p.step_rad=0; CHECK_THROWS_AS(Controller{p},std::invalid_argument);
  Controller c; double now=2; CHECK_FALSE(c.StartMove(0,kDeg,Healthy(kOrigin,now),now)); Init(c,now);
  CHECK_FALSE(c.StartMove(5,kDeg,Healthy(kOrigin,now),now));
  CHECK_FALSE(c.StartMove(0,2*kDeg,Healthy(kOrigin,now),now));
  CHECK(c.status().q==kOrigin);
}
#include "runtime/arm_startup_check.h"
TEST_CASE("startup waits for actual home settling and retains guards") {
  StartupCheck gate(10,kOrigin);
  auto s=Healthy(kOrigin,10.01); s.q[1]+=2*kDeg; s.dq[1]=kDeg;
  CHECK_FALSE(gate.Update(s,10.01)); CHECK_FALSE(gate.fault());
  for(int n=2;n<115;++n) gate.Update(Healthy(kOrigin,10+n*.01),10+n*.01);
  CHECK(gate.reason()=="startup_settled");
  for(int mode=0;mode<6;++mode) {
    StartupCheck bad(10,kOrigin); auto f=Healthy(kOrigin,10.01);
    if(mode==0) f.dq[0]=1081*kDeg;
    if(mode==1) f.q[0]+=361*kDeg;
    if(mode==2) f.stamp_s[0]=9;
    if(mode==3) f.fault_bits[0]=1;
    if(mode==4) f.mode_state[0]=0;
    if(mode==5) f.q[0]=std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(bad.Update(f,10.01)); CHECK(bad.fault());
    CHECK_FALSE(bad.Update(Healthy(kOrigin,10.02),10.02));
  }
}
TEST_CASE("startup requires fresh continuous feedback and bounded waiting") {
  StartupCheck frozen(10,kOrigin); auto s=Healthy(kOrigin,10.01);
  for(int n=1;n<10;++n) CHECK_FALSE(frozen.Update(s,10+n*.01));
  CHECK_FALSE(frozen.Update(s,10.12)); CHECK(frozen.fault());
  StartupCheck gap(10,kOrigin); gap.Update(Healthy(kOrigin,10.01),10.01);
  CHECK_FALSE(gap.Update(Healthy(kOrigin,10.1),10.1)); CHECK(gap.fault());
  StartupCheck timeout(10,kOrigin);
  for(int n=1;n<1505 && !timeout.fault();++n) {
    auto f=Healthy(kOrigin,10+n*.01); f.q[2]+=(n%2 ? 2 : -2)*kDeg;
    CHECK_FALSE(timeout.Update(f,10+n*.01));
  }
  CHECK(timeout.reason()=="startup_settle_timeout");
}

TEST_CASE("1080 degree feedback threshold is shared by startup and controller") {
  for(double speed : {11., 1079., 1080., -1080.}) {
    auto f=Healthy(kOrigin,10.01); f.dq[1]=speed*kDeg;
    Controller c;
    CHECK(c.Initialize(f,10.01,kOrigin));
    CHECK_FALSE(c.status().fault);
    StartupCheck startup(10,kOrigin);
    CHECK_FALSE(startup.Update(f,10.01));
    CHECK_FALSE(startup.fault());
  }
  for(double speed : {1080.01, -1080.01}) {
    auto f=Healthy(kOrigin,10.01); f.dq[1]=speed*kDeg;
    Controller c;
    CHECK_FALSE(c.Initialize(f,10.01,kOrigin));
    CHECK(c.status().reason=="feedback_speed:J2");
    StartupCheck startup(10,kOrigin);
    CHECK_FALSE(startup.Update(f,10.01));
    CHECK(startup.reason()=="startup_feedback_speed:J2");
  }
}

TEST_CASE("startup 360 degree error boundary and measured J2 pose") {
  for(double error:{-360.,-180.,-49.8759469422,180.,360.}) {
    StartupCheck gate(10,Joints{}); auto f=Healthy(Joints{},10.01);
    f.q[1]=error*kDeg; f.dq[1]=67.0406221378*kDeg;
    CHECK_FALSE(gate.Update(f,10.01)); CHECK_FALSE(gate.fault());
    CHECK(gate.reason()=="startup_settling");
  }
  for(double error:{-360.01,360.01}) {
    StartupCheck gate(10,Joints{}); auto f=Healthy(Joints{},10.01); f.q[1]=error*kDeg;
    CHECK_FALSE(gate.Update(f,10.01)); CHECK(gate.fault());
    CHECK(gate.reason()=="startup_tracking_error:J2");
  }
  // 重现已记录的J2/J3初始偏差，随后用合成到位反馈验证状态门。
  StartupCheck gate(10,kOrigin);
  for(int n=1;n<=165;++n) {
    double now=10+n*.01; auto f=Healthy(kOrigin,now);
    if(n<50) { f.q[1]-=49.8759469422*kDeg; f.q[2]-=16.1680976617*kDeg; f.dq[1]=67.0406221378*kDeg; }
    gate.Update(f,now); REQUIRE_FALSE(gate.fault());
    if(n<50) CHECK(gate.reason()=="startup_settling");
  }
  CHECK(gate.reason()=="startup_settled");
  CHECK(Policy{}.max_tracking_rad==8*kDeg);
}

TEST_CASE("position window uses each axis fresh stamps and strict span") {
  PositionWindow window; Policy p;
  for(int n=1;n<=120;++n) {
    auto f=Healthy(kOrigin,10+n*.01); f.stamp_s[4]=10.009;
    CHECK_FALSE(window.Update(f,p));
  }
  window.Reset();
  for(int n=1;n<=150;++n) {
    auto f=Healthy(Joints{},10+n*.01); f.q[0]=(n%2 ? 0 : p.settle_span_rad);
    CHECK_FALSE(window.Update(f,p));
  }
  window.Reset();
  for(int n=1;n<=120;++n) {
    auto f=Healthy(kOrigin,10+n*.01); f.q[2]+=(n%2 ? .7 : -.7)*kDeg; f.dq.fill(14*kDeg);
    const bool stable=window.Update(f,p); if(n<100) CHECK_FALSE(stable); if(n>105) CHECK(stable);
  }
}
