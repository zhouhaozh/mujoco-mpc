// 仅以 fake 验证共享的派发路径；不创建 Robot、不启动任何真实或模拟服务。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <limits>
#include <memory>
#include <stdexcept>

#include "runtime/x4_sdk_backend.h"

using namespace sckd::io;

namespace {
constexpr Time kNow{10., Clock::kMonotonic};

X4SdkJointEncoding IdentityEncoding() {
  X4SdkJointEncoding encoding;
  encoding.sign.fill(1.0);  // 测试场景显式指定，不作为生产缺省。
  return encoding;
}

Command ValidCommand(std::uint64_t sequence = 1) {
  Command c;
  c.sequence = sequence;
  c.plan_id = 42;
  c.issued = {9., Clock::kMonotonic};
  c.expires = {11., Clock::kMonotonic};
  c.frame_id = "sdk_model";
  c.body_twist = {.1, -.2, .3};
  c.q = {.1, -.2, .3, -.4, .5};
  c.dq.fill(.1);
  c.kp.fill(20.);
  c.kd.fill(2.);
  return c;
}

class FakeEndpoint final : public testing::X4SdkEndpoint {
 public:
  Capabilities capabilities() const override { return {false, true, "sdk_model"}; }
  X4SdkJointEncoding joint_encoding() const override { return encoding; }
  testing::X4SdkEndpointStatus status() const override { return state; }
  void Submit(const Command& c) override {
    ++writes;
    submitted = c;
    if (throw_on_submit) throw std::runtime_error("test transport failure");
  }
  Observation Snapshot(Time now) override {
    ++reads;
    Observation out;
    out.received = now;
    out.frame_id = "sdk_model";
    out.ready = state.ready;
    out.arm_degraded = state.arm_degraded;
    out.manual_takeover = state.manual_takeover;
    out.base_available = true;
    out.base_source = Source::kCommandIntegrated;
    out.base_stamp.clock = Clock::kMonotonic;
    out.joints[0].stamp = {8.25, Clock::kMonotonic};
    out.joints[0].stamp_known = true;
    out.joints[0].available = true;
    out.coherent_snapshot = true;  // wrapper 必须纠正为非整机一致快照。
    return out;
  }
  testing::X4SdkEndpointStatus state{true, false, false};
  int writes = 0, reads = 0;
  bool throw_on_submit = false;
  X4SdkJointEncoding encoding = IdentityEncoding();
  Command submitted;
};

struct Fixture {
  Fixture() {
    auto endpoint = std::make_unique<FakeEndpoint>();
    fake = endpoint.get();
    backend = testing::WrapX4SdkEndpoint(std::move(endpoint));
  }
  void RejectWithoutWrite(const Command& command, const std::string& reason,
                          Time now = kNow) {
    const int before = fake->writes;
    const auto result = backend->Write(command, now);
    CHECK_FALSE(result.submitted);
    CHECK(result.reason == reason);
    CHECK(fake->writes == before);
  }
  FakeEndpoint* fake = nullptr;
  std::unique_ptr<RobotBackend> backend;
};
}  // namespace

TEST_CASE("exported SDK preflight uses the complete dispatch validation without a backend") {
  auto c = ValidCommand(8);
  const Capabilities caps{false, false, "sdk_model"};
  const auto encoding = IdentityEncoding();
  CHECK(ValidateX4SdkCommand(c, kNow, caps, encoding, 7).submitted);
  CHECK(ValidateX4SdkCommand(c, kNow, caps, encoding, 8).reason == "nonincreasing_sequence");
  c.body_twist[0] = static_cast<double>(std::numeric_limits<float>::max()) / 100.;
  CHECK(ValidateX4SdkCommand(c, kNow, caps, encoding).reason == "float_overflow");
  c = ValidCommand(8);
  c.kp[1] = 1e20;
  CHECK(ValidateX4SdkCommand(c, kNow, caps, encoding).reason == "gain_out_of_range");
  c = ValidCommand(8);
  c.kd[4] = 0;
  CHECK(ValidateX4SdkCommand(c, kNow, caps, encoding).reason == "nonpositive_gain");
  c = ValidCommand(8);
  c.body_twist[0] = static_cast<double>(std::numeric_limits<float>::max()) / 1000.;
  const auto edge = ValidateX4SdkCommand(c, kNow, caps, encoding);
  const float facade_mmps = static_cast<float>(c.body_twist[0]) * 1000.f;
  CHECK((!edge.submitted || std::isfinite(facade_mmps)));
}

TEST_CASE("SDK preflight rejects invalid commands before any endpoint write") {
  Fixture f;
  auto c = ValidCommand();
  SUBCASE("zero sequence") { c.sequence = 0; f.RejectWithoutWrite(c, "invalid_sequence"); }
  SUBCASE("frame mapping is explicit") { c.frame_id = "mpc_model"; f.RejectWithoutWrite(c, "frame_mismatch"); }
  SUBCASE("simulation clock is not the SDK clock") {
    c.issued.clock = c.expires.clock = Clock::kSimulation;
    f.RejectWithoutWrite(c, "clock_mismatch", {10., Clock::kSimulation});
  }
  SUBCASE("mismatched issued clock") { c.issued.clock = Clock::kSimulation; f.RejectWithoutWrite(c, "clock_mismatch"); }
  SUBCASE("expiry boundary") { c.expires.seconds = 10.; f.RejectWithoutWrite(c, "expired"); }
  SUBCASE("future command") { c.issued.seconds = 10.5; f.RejectWithoutWrite(c, "not_yet_valid"); }
  SUBCASE("invalid lifetime") { c.expires.seconds = 8.; f.RejectWithoutWrite(c, "invalid_time"); }
  SUBCASE("invalid now") { f.RejectWithoutWrite(c, "invalid_time", {std::numeric_limits<double>::quiet_NaN(), Clock::kMonotonic}); }
  SUBCASE("NaN joint") { c.q[4] = std::numeric_limits<double>::quiet_NaN(); f.RejectWithoutWrite(c, "nonfinite_command"); }
  SUBCASE("infinite base") { c.body_twist[1] = std::numeric_limits<double>::infinity(); f.RejectWithoutWrite(c, "nonfinite_command"); }
  SUBCASE("unsupported feedforward") { c.tau_ff[3] = 1e-20; f.RejectWithoutWrite(c, "unsupported_tau_ff"); }
  SUBCASE("zero kp does not silently select SDK defaults") { c.kp[0] = 0; f.RejectWithoutWrite(c, "nonpositive_gain"); }
  SUBCASE("zero kd does not silently select SDK defaults") { c.kd[0] = 0; f.RejectWithoutWrite(c, "nonpositive_gain"); }
  SUBCASE("negative gain") { c.kp[2] = -1; f.RejectWithoutWrite(c, "negative_gain"); }
  SUBCASE("finite gain exceeds SDK model encoding range") { c.kp[1] = 1e20; f.RejectWithoutWrite(c, "gain_out_of_range"); }
  SUBCASE("finite position exceeds MIT encoding range") { c.q[0] = 1000.; f.RejectWithoutWrite(c, "position_encoding_range"); }
  SUBCASE("finite velocity exceeds MIT encoding range") { c.dq[0] = 1000.; f.RejectWithoutWrite(c, "velocity_encoding_range"); }
  SUBCASE("float gain underflow cannot become SDK default") { c.kp[1] = std::numeric_limits<double>::denorm_min(); f.RejectWithoutWrite(c, "gain_float_underflow"); }
  SUBCASE("finite double overflows joint float") { c.q[0] = std::numeric_limits<double>::max(); f.RejectWithoutWrite(c, "float_overflow"); }
  SUBCASE("SI conversion overflows SDK mm per second") {
    c.body_twist[0] = static_cast<double>(std::numeric_limits<float>::max()) / 100.;
    f.RejectWithoutWrite(c, "float_overflow");
  }
  // 被拒绝指令不消耗序号；修正后仍可提交同一 sequence。
  CHECK(f.backend->Write(ValidCommand(), kNow).submitted);
  CHECK(f.fake->writes == 1);
}

TEST_CASE("SDK encoding bounds are checked after explicit model to motor conversion") {
  Fixture f;
  auto c = ValidCommand();
  SUBCASE("default unverified context is rejected") {
    f.fake->encoding = {};
    f.RejectWithoutWrite(c, "invalid_joint_encoding_context");
  }
  SUBCASE("offset permits model angle outside raw motor range") {
    // J1 MIT范围约±12.57；模型角13经offset2变成电机角11，应接受。
    f.fake->encoding.offset_rad[0] = 2.;
    c.q[0] = 13.;
    CHECK(f.backend->Write(c, kNow).submitted);
    CHECK(f.fake->writes == 1);
  }
  SUBCASE("offset can invalidate model angle inside raw motor range") {
    f.fake->encoding.offset_rad[0] = -2.;
    c.q[0] = 12.;  // 电机角14，不能静默钳到12.57。
    f.RejectWithoutWrite(c, "position_encoding_range");
  }
  SUBCASE("negative joint direction is applied before range checks") {
    f.fake->encoding.sign[0] = -1.;
    f.fake->encoding.offset_rad[0] = 2.;
    c.q[0] = 13.;  // motor=-11，仍可编码。
    c.dq[0] = 35.;
    CHECK(f.backend->Write(c, kNow).submitted);
    CHECK(f.fake->writes == 1);
  }
  SUBCASE("nonfinite offset is rejected") {
    f.fake->encoding.offset_rad[0] = std::numeric_limits<double>::quiet_NaN();
    f.RejectWithoutWrite(c, "invalid_joint_encoding_context");
  }
}

TEST_CASE("SDK rejects duplicate and out of order sequence after successful submission") {
  Fixture f;
  CHECK(f.backend->Write(ValidCommand(10), kNow).submitted);
  REQUIRE(f.fake->writes == 1);
  f.RejectWithoutWrite(ValidCommand(10), "nonincreasing_sequence");
  f.RejectWithoutWrite(ValidCommand(9), "nonincreasing_sequence");
  CHECK(f.backend->Write(ValidCommand(11), kNow).submitted);
  CHECK(f.fake->writes == 2);
  CHECK(f.fake->submitted.plan_id == 42);
  CHECK(f.fake->submitted.body_twist[1] == doctest::Approx(-.2));
}

TEST_CASE("SDK status gating precedes both arm and chassis submission") {
  Fixture f;
  const auto c = ValidCommand();
  SUBCASE("not ready") { f.fake->state.ready = false; f.RejectWithoutWrite(c, "backend_not_ready"); }
  SUBCASE("degraded arm") { f.fake->state.arm_degraded = true; f.RejectWithoutWrite(c, "arm_degraded"); }
  SUBCASE("manual chassis takeover blocks whole body command") { f.fake->state.manual_takeover = true; f.RejectWithoutWrite(c, "manual_takeover"); }
  f.fake->state = {true, false, false};
  CHECK(f.backend->Write(c, kNow).submitted);
  CHECK(f.fake->writes == 1);
}

TEST_CASE("possible partial submission consumes sequence even when transport throws") {
  Fixture f;
  f.fake->throw_on_submit = true;
  const auto result = f.backend->Write(ValidCommand(5), kNow);
  CHECK_FALSE(result.submitted);
  CHECK(result.reason.find("submission_exception:") == 0);
  CHECK(f.fake->writes == 1);
  f.fake->throw_on_submit = false;
  f.RejectWithoutWrite(ValidCommand(5), "nonincreasing_sequence");
  CHECK(f.backend->Write(ValidCommand(6), kNow).submitted);
}

TEST_CASE("SDK reads preserve source timestamps and never invent a base timestamp") {
  Fixture f;
  const auto first = f.backend->Read(kNow);
  CHECK(first.sequence == 1);
  CHECK(first.frame_id == "sdk_model");
  CHECK_FALSE(first.coherent_snapshot);
  CHECK_FALSE(first.base_stamp_known);
  CHECK(first.base_stamp.seconds == 0);
  CHECK(first.base_source == Source::kCommandIntegrated);
  CHECK(first.joints[0].stamp_known);
  CHECK(first.joints[0].stamp.seconds == doctest::Approx(8.25));
  CHECK(first.joints[0].stamp.clock == Clock::kMonotonic);
  CHECK(f.backend->Read(kNow).sequence == 2);
  CHECK(f.fake->reads == 2);
  CHECK(f.fake->writes == 0);
  CHECK_THROWS_AS(f.backend->Read({10., Clock::kSimulation}), std::invalid_argument);
  CHECK(f.fake->reads == 2);
}

TEST_CASE("SDK wrapper rejects a missing endpoint") {
  CHECK_THROWS_AS(testing::WrapX4SdkEndpoint(nullptr), std::invalid_argument);
}
