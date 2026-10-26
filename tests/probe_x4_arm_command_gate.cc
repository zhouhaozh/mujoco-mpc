// Ubuntu 离线测试：实际 SDK 配置/构造/目标缓存，设备入口全部由强符号替换。
// 不链接测试版生产代码、不启动线程/CAN；另以 seccomp 禁止创建任何 socket。
#include "runtime/x4_arm_test_backend.h"
#include "arm/server.h"

#include <dlfcn.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using sckd::arm::ArmServer;
using sckd::arm::ArmStatus;
using sckd::arm::JointCmd;
std::vector<std::string> events;
bool enabled = false, running = false, seeded = false;
int start_calls = 0, enable_calls = 0, write_calls = 0;
std::vector<sckd::arm::ArmStatus> snapshots;
size_t snapshot_index = 0;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <typename Function> Function Original(const char* symbol) {
  void* fn = dlsym(RTLD_NEXT, symbol);
  if (!fn) throw std::runtime_error(dlerror());
  return reinterpret_cast<Function>(fn);
}
[[noreturn]] void ForbiddenCan() {
  const char message[] = "FAIL: CAN operation reached in offline probe\n";
  ::write(STDERR_FILENO, message, sizeof(message) - 1);
  std::abort();
}
void BlockSockets() {
  sock_filter instructions[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_socket, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)};
  sock_fprog program{static_cast<unsigned short>(sizeof(instructions) /
                                                sizeof(instructions[0])), instructions};
  Require(prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0, "no_new_privs failed");
  Require(prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0,
          "socket-blocking seccomp failed");
}
}  // namespace

// ELF 符号抢占：实际 SDK 虚表解析到这些测试实现，绝不调用真实 Start/Stop。
bool sckd::arm::ArmServer::Start(std::string*) {
  ++start_calls; running = true; events.emplace_back("start_mock"); return true;
}
void sckd::arm::ArmServer::Stop() { running = false; }
sckd::arm::ArmStatus sckd::arm::ArmServer::Snapshot() const {
  if(!snapshots.empty()) return snapshots[std::min(snapshot_index++,snapshots.size()-1)];
  ArmStatus state;
  for (bool& open : state.can_open) open = true;
  return state;
}
// IsRunning/SetEnabled 是头文件内联虚方法，使用 ABI 符号名避免重定义。
extern "C" bool MockIsRunning(const ArmServer*)
    asm("_ZNK4sckd3arm9ArmServer9IsRunningEv");
extern "C" bool MockIsRunning(const ArmServer*) { return running; }
extern "C" void MockSetEnabled(ArmServer*, bool)
    asm("_ZN4sckd3arm9ArmServer10SetEnabledEb");
extern "C" void MockSetEnabled(ArmServer* server, bool value) {
  Require(seeded, "target channel enabled before first valid target");
  Require(value, "test unexpectedly disabled channel (SDK would return home)");
  ++enable_calls; enabled = value; events.emplace_back("enable_true");
  Original<void (*)(ArmServer*, bool)>("_ZN4sckd3arm9ArmServer10SetEnabledEb")(
      server, value);  // 实际 SDK 的纯内存原子写。
}
void sckd::arm::ArmServer::SetJointTargets(const JointCmd* target) {
  Original<void (*)(ArmServer*, const JointCmd*)>(
      "_ZN4sckd3arm9ArmServer15SetJointTargetsEPKNS0_8JointCmdE")(this, target);
  JointCmd readback[5];
  LastJointCmds(readback);  // 实际 SDK seqlock 读回，不是仿造命令缓存。
  for (int j = 0; j < 5; ++j)
    Require(readback[j].pos == target[j].pos && readback[j].vel == target[j].vel &&
                readback[j].kp == target[j].kp && readback[j].kd == target[j].kd,
            "SDK target cache readback mismatch");
  seeded = true; ++write_calls; events.emplace_back("target_seeded");
}
bool sckd::hal::CanBus::Open(const Options&, std::string*) { ForbiddenCan(); }
bool sckd::hal::CanBus::Send(const CanFdFrame&) { ForbiddenCan(); }
int sckd::hal::CanBus::Recv(CanFdFrame*, int, int64_t*) { ForbiddenCan(); }
void sckd::hal::CanBus::Close() {}  // 真实构造的 fd 从未打开，析构仅关闭空对象。

int main(int argc, char** argv) {
  try {
    Require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--expect-missing"),
            "usage: probe CONFIG [--expect-missing]");
    const bool expect_missing = argc == 3;
    BlockSockets();
    using namespace sckd::io::axischeck;
    SdkArmBackend backend(argv[1]);
    std::string error;
    Require(backend.StartHardware(&error), "mock startup failed");
    Require(start_calls == 1 && write_calls == 0 && enable_calls == 0,
            "startup changed before first Write");
    Output output;
    output.q = backend.setup().home;
    Require(!backend.Write(output, &error), "uninitialized output accepted");
    output.initialized = true;
    output.q[0] = std::numeric_limits<double>::quiet_NaN();
    Require(!backend.Write(output, &error), "NaN output accepted");
    Require(write_calls == 0 && enable_calls == 0, "invalid write opened channel");
    output.q = backend.setup().home;
    Require(backend.Write(output, &error), "valid home Write failed");
    if (expect_missing) {
      Require(!enabled && enable_calls == 0, "baseline does not reproduce missing enable");
    } else {
      Require(enabled && enable_calls == 1, "valid target did not enable SDK channel");
      Require(events.size() == 3 && events[1] == "target_seeded" &&
                  events[2] == "enable_true", "wrong seed/enable order");
    }
    output.q[0] += 10 * kDeg;
    Require(backend.Write(output, &error), "J1 +10deg Write failed");
    Require(std::abs(backend.LastTarget()[0] - output.q[0]) < 1e-6,
            "J1 target missing from real SDK cache");
    Require(enable_calls == (expect_missing ? 0 : 1), "channel enabled more than once");
    const int count = write_calls;
    output.q[1] = std::numeric_limits<double>::quiet_NaN();
    Require(!backend.Write(output, &error) && write_calls == count,
            "invalid later target altered SDK cache");
    // 真正Read适配器，只有设备/快照入口被替换；socket仍被seccomp禁止。
    ArmStatus valid;
    timespec ts{}; clock_gettime(CLOCK_MONOTONIC,&ts);
    const auto stamp=static_cast<int64_t>(ts.tv_sec)*1000000+ts.tv_nsec/1000-100;
    for(bool& open:valid.can_open) open=true;
    for(int j=0;j<5;++j) {
      valid.joints[j].pos=.1f*(j+1); valid.joints[j].stamp_us=stamp;
      valid.joints[j].mode_state=2;
    }
    auto missing=valid; missing.joints[0]={};
    auto ReadPlan=[&](std::initializer_list<ArmStatus> plan) {
      snapshots=plan; snapshot_index=0; return backend.Read();
    };
    auto f=ReadPlan({missing,valid});
    Require(f.snapshot_attempts==2 && f.sample.mode_state[0]==2 &&
            f.sample.stamp_s[0]==static_cast<double>(stamp)*1e-6 &&
            f.sample.q[0]==valid.joints[0].pos,"missing slot retry did not preserve SDK sample");
    f=ReadPlan({missing,missing,missing,valid});
    Require(f.snapshot_attempts==3 && f.sample.stamp_s[0]==0 && f.raw_mode_state[0]==0,
            "persistent missing slot bypassed retry bound");
    auto bad=missing; bad.joints[1].fault_bits=1;
    f=ReadPlan({bad,valid});
    Require(f.snapshot_attempts==1 && f.sample.fault_bits[1]==1,"retry swallowed motor fault");
    bad=missing; bad.joints[1].mode_state=0;
    f=ReadPlan({bad,valid});
    Require(f.snapshot_attempts==1 && f.sample.mode_state[1]==0,"retry swallowed motor mode change");
    bad=valid; bad.joints[0].stamp_us=stamp-200000;
    f=ReadPlan({missing,bad,valid});
    Require(f.snapshot_attempts==2 && f.sample.stamp_s[0]==static_cast<double>(stamp-200000)*1e-6,
            "retry replaced stale recovery timestamp");
    bad=missing; bad.can_open[1]=false;
    f=ReadPlan({bad,valid});
    Require(f.snapshot_attempts==1 && !f.can_open[1],"retry swallowed CAN failure");
    bad=missing; bad.joints[1].speed=std::numeric_limits<float>::infinity();
    f=ReadPlan({bad,valid});
    Require(f.snapshot_attempts==1 && !std::isfinite(f.sample.dq[1]),"retry swallowed nonfinite feedback");
    snapshots.clear();
    backend.Stop();
    Require(enable_calls == (expect_missing ? 0 : 1), "Stop toggled target gate");
    Require(!backend.Write(output, &error), "Write accepted after Stop");
    std::cout << "{\"result\":\"pass\",\"baseline_missing_enable\":"
              << (expect_missing ? "true" : "false")
              << ",\"actual_sdk_cache_checked\":true,\"socket_creation_blocked\":true,"
              << "\"snapshot_retry_cases\":7,\"start_mock_calls\":" << start_calls << ",\"target_calls\":"
              << write_calls << ",\"enable_calls\":" << enable_calls << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
