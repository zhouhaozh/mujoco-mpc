#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "runtime/robot_io_log.h"
#include "runtime/robot_io_mapping.h"
#include <limits>
#include <unistd.h>
using namespace sckd::io;
static Command Sample() {
  Command c;c.sequence=1;c.issued={1,Clock::kSimulation};c.expires={1.004,Clock::kSimulation};
  c.kp.fill(10);c.kd.fill(1);c.q={.1,.2,.3,.4,.5};return c;
}
TEST_CASE("deadline and capability gates reject before side effects") {
  const Capabilities caps{false,false,"mpc_model"};auto c=Sample();
  CHECK(Validate(c,{1.003,Clock::kSimulation},caps).submitted);
  CHECK(Validate(c,{1.004,Clock::kSimulation},caps).reason=="expired");
  CHECK(Validate(c,{.9,Clock::kSimulation},caps).reason=="not_yet_valid");
  CHECK(Validate(c,{1,Clock::kMonotonic},caps).reason=="clock_mismatch");
  c.tau_ff[2]=1e-12;CHECK(Validate(c,c.issued,caps).reason=="unsupported_tau_ff");
  CHECK(Validate(c,c.issued,{true,false,"mpc_model"}).submitted);
  c.tau_ff[2]=0;c.frame_id="sdk_model";CHECK(Validate(c,c.issued,caps).reason=="frame_mismatch");
  c=Sample();c.body_twist[1]=std::numeric_limits<double>::quiet_NaN();
  CHECK(Validate(c,c.issued,caps).reason=="nonfinite_command");
  c=Sample();c.kd[0]=-1;CHECK(Validate(c,c.issued,caps).reason=="negative_gain");
}
TEST_CASE("log roundtrip preserves exact commands and unknown timestamp provenance") {
  auto c=Sample();c.tau_ff[4]=2.5;
  auto j=ToJson(c);auto copy=CommandFromJson(Json::parse(j.dump()));
  CHECK(ToJson(copy)==j);
  j["q"]={1,2};CHECK_THROWS(CommandFromJson(j));
  j=ToJson(c);j["sequence"]=-1;CHECK_THROWS(CommandFromJson(j));
  j=ToJson(c);j["issued"]["clock"]="camera_ptp";CHECK_THROWS(CommandFromJson(j));
  Observation o;o.base_available=true;o.base_source=Source::kCommandIntegrated;
  const auto state=ObservationToJson(o);
  CHECK(state["base_stamp"].is_null());CHECK(state["base_source"]=="command_integrated");
  CHECK(state["coherent_snapshot"]==false);
  const auto dir=std::filesystem::temp_directory_path()/("robot-io-"+std::to_string(getpid()));
  std::filesystem::create_directories(dir);const auto path=(dir/"commands.jsonl").string();
  {
    CommandLog log(path,{{"test",true}});log.Write(c,o,{true,"submitted"});
    CHECK_THROWS(CommandLog(path,{}));
  }
  auto stream=ReadCommands(path);REQUIRE(stream.commands.size()==1);CHECK(ToJson(stream.commands[0])==ToJson(c));
  {std::ofstream f(path,std::ios::app);f<<Json({{"schema","X4_ROBOT_IO/1"},{"type","sample"},
    {"command",ToJson(c)},{"result",{{"submitted",true}}}}).dump()<<'\n';}
  CHECK_THROWS(ReadCommands(path));
  std::filesystem::remove(path);std::filesystem::remove(dir);
}
TEST_CASE("explicit model map transforms position velocity and torque consistently") {
  JointFrameMap m;m.from_frame="mpc_model";m.to_frame="sdk_model";m.version="synthetic-test";
  m.sign={-1,1,-1,1,1};m.offset_rad={.2,0,0,0,0};
  auto c=Sample();c.dq[0]=2;c.tau_ff[0]=3;
  auto mapped=m.Apply(c);
  CHECK(mapped.q[0]==doctest::Approx(.1));CHECK(mapped.dq[0]==-2);CHECK(mapped.tau_ff[0]==-3);
  CHECK(mapped.kp==c.kp);CHECK(mapped.body_twist==c.body_twist);CHECK(mapped.frame_id=="sdk_model");
  CHECK_THROWS(m.Apply(mapped));
}
