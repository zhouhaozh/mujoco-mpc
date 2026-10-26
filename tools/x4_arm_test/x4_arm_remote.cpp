// 隔离真机包的原生入口：inspect只读，prepare只准备机械臂CAN，run交给控制器确认。
// 不使用shell、不包含密码、不发送START-HOME或任何电机目标。
#include <nlohmann/json.hpp>

#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace {
constexpr std::array<const char*, 3> kChannels{"can4", "can5", "can6"};

class Fd {
 public:
  explicit Fd(int value = -1) : value_(value) {}
  ~Fd() { if (value_ >= 0) ::close(value_); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return value_; }
 private:
  int value_;
};

std::runtime_error SystemError(const std::string& action) {
  return std::runtime_error(action + ": " + std::strerror(errno));
}

std::vector<char*> Arguments(std::vector<std::string>& values) {
  std::vector<char*> argv;
  for (auto& value : values) argv.push_back(value.data());
  argv.push_back(nullptr);
  return argv;
}

// stdout捕获仅用于ip JSON；其他命令继承真实终端，sudo密码不经本程序记录。
std::string Run(std::vector<std::string> command, bool capture = false) {
  if (command.empty()) throw std::logic_error("empty command");
  int pipefd[2] = {-1, -1};
  if (capture && ::pipe2(pipefd, O_CLOEXEC) < 0) throw SystemError("pipe2");
  const pid_t child = ::fork();
  if (child < 0) {
    const int saved = errno;
    if (capture) { ::close(pipefd[0]); ::close(pipefd[1]); }
    errno = saved;
    throw SystemError("fork");
  }
  if (child == 0) {
    if (capture) {
      ::close(pipefd[0]);
      if (::dup2(pipefd[1], STDOUT_FILENO) < 0) _exit(126);
      ::close(pipefd[1]);
    }
    auto argv = Arguments(command);
    ::execvp(argv[0], argv.data());
    ::dprintf(STDERR_FILENO, "无法执行 %s: %s\n", command[0].c_str(), std::strerror(errno));
    _exit(127);
  }
  std::string output;
  bool read_failed = false, overflow = false;
  if (capture) {
    ::close(pipefd[1]);
    Fd reader(pipefd[0]);
    char buffer[4096];
    for (;;) {
      const ssize_t count = ::read(reader.get(), buffer, sizeof(buffer));
      if (count == 0) break;
      if (count < 0) {
        if (errno == EINTR) continue;
        read_failed = true;
        ::kill(child, SIGTERM);
        break;
      }
      if (output.size() + static_cast<std::size_t>(count) > 1024 * 1024) {
        overflow = true;
        ::kill(child, SIGTERM);
        break;
      }
      output.append(buffer, static_cast<std::size_t>(count));
    }
  }
  int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) throw SystemError("waitpid");
  }
  if (read_failed || overflow)
    throw std::runtime_error(command[0] + (overflow ? "输出过长" : "输出读取失败"));
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error(command[0] + "执行失败，状态=" + std::to_string(status));
  return output;
}

struct Options {
  std::string action = "inspect", step = "30", test_mode = "direction";
  bool help = false;
};

Options Parse(int argc, char** argv) {
  Options options;
  bool have_action = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") options.help = true;
    else if (key == "--step-deg" || key == "--test-mode") {
      if (++i == argc) throw std::invalid_argument("缺少参数: " + key);
      (key == "--step-deg" ? options.step : options.test_mode) = argv[i];
    } else if (!have_action && (key == "inspect" || key == "prepare" || key == "run")) {
      options.action = key;
      have_action = true;
    } else throw std::invalid_argument("未知参数: " + key);
  }
  if (options.step != "1" && options.step != "10" && options.step != "30")
    throw std::invalid_argument("--step-deg须为1、10或30");
  if (options.test_mode != "direction" && options.test_mode != "tracking")
    throw std::invalid_argument("--test-mode须为direction或tracking");
  return options;
}

Json CanInfo(const std::string& name) {
  const auto result = Json::parse(Run({"ip", "-details", "-j", "link", "show", name}, true));
  if (!result.is_array() || result.size() != 1 || !result[0].is_object())
    throw std::runtime_error(name + "的ip JSON格式不符");
  const auto& state = result[0];
  if (state.contains("ifname") && state.at("ifname") != name)
    throw std::runtime_error(name + "返回了其他接口");
  return state;
}

bool Contains(const Json& value, const std::string& field, const std::string& expected) {
  const auto it = value.find(field);
  if (it == value.end() || !it->is_array()) return false;
  for (const auto& item : *it) if (item == expected) return true;
  return false;
}

bool Configured(const Json& state) {
  try {
    const auto& data = state.at("linkinfo").at("info_data");
    return Contains(state, "flags", "UP") &&
           data.at("bittiming").at("bitrate") == 1000000 &&
           data.at("data_bittiming").at("bitrate") == 2000000 &&
           Contains(data, "ctrlmode", "FD") && Contains(data, "ctrlmode", "ONE-SHOT") &&
           data.at("state") == "ERROR-ACTIVE";
  } catch (const Json::exception&) { return false; }
}

void CheckProcesses() {
  // 已知控制程序的辅助互斥检查；SocketCAN本身不提供跨应用独占。
  const std::array<std::string, 6> blocked{
      "robotd", "court_run", "arm_test", "arm_pose", "arm_joint_tune", "x4_arm_axis_test"};
  const auto self = std::to_string(::getpid());
  std::error_code error;
  fs::directory_iterator entries("/proc", fs::directory_options::skip_permission_denied, error);
  if (error) throw std::runtime_error("无法检查/proc: " + error.message());
  for (const auto& entry : entries) {
    const auto pid = entry.path().filename().string();
    if (pid == self || pid.empty() || pid.find_first_not_of("0123456789") != std::string::npos) continue;
    std::string name, executable;
    std::ifstream comm(entry.path() / "comm");
    std::getline(comm, name);
    std::ifstream cmdline(entry.path() / "cmdline", std::ios::binary);
    std::getline(cmdline, executable, '\0');
    const auto basename = fs::path(executable).filename().string();
    for (const auto& candidate : blocked) {
      // Linux comm最多15字节；cmdline可能因权限而不可读。
      if (basename == candidate || name == candidate.substr(0, 15))
        throw std::runtime_error("已有控制进程，先由操作者处理：" + pid + " " + name);
    }
  }
}

class EchoGuard {
 public:
  EchoGuard() {
    if (::tcgetattr(STDIN_FILENO, &original_) < 0) throw SystemError("tcgetattr");
    auto updated = original_;
    updated.c_lflag &= ~ECHO;
    if (::tcsetattr(STDIN_FILENO, TCSANOW, &updated) < 0) throw SystemError("tcsetattr");
    active_ = true;
  }
  ~EchoGuard() { if (active_) ::tcsetattr(STDIN_FILENO, TCSANOW, &original_); }
 private:
  termios original_{};
  bool active_ = false;
};

std::string RecordName() {
  const auto now = std::chrono::system_clock::now();
  const auto epoch_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  const time_t seconds = static_cast<time_t>(epoch_us / 1000000);
  std::tm local{};
  if (!::localtime_r(&seconds, &local)) throw std::runtime_error("localtime_r失败");
  std::ostringstream result;
  result << std::put_time(&local, "%Y%m%d_%H%M%S_") << std::setfill('0') << std::setw(6)
         << epoch_us % 1000000;
  return result.str();
}

int Main(int argc, char** argv) {
  const auto options = Parse(argc, argv);
  if (options.help) {
    std::cout << "x4_arm_remote [inspect|prepare|run] [--step-deg 1|10|30] [--test-mode direction|tracking]\n"
      "inspect只读；prepare仅配置can4–6；run仍须现场输入START-HOME。\n";
    return 0;
  }
  const auto root = fs::canonical("/proc/self/exe").parent_path().parent_path();
  const auto binary = root / "bin/x4_arm_axis_test";
  const auto config = root / "config/robot_8w4s.ini";
  if (::setenv("LD_LIBRARY_PATH", (root / "lib").c_str(), 1) < 0) throw SystemError("setenv");
  const std::vector<std::string> inspect{
      binary.string(), "--config", config.string(), "--step-deg", options.step,
      "--test-mode", options.test_mode};
  if (options.action == "inspect") {
    Run(inspect);
    for (const auto* channel : kChannels) std::cout << CanInfo(channel).dump() << '\n';
    return 0;
  }
  if (!::isatty(STDIN_FILENO) || !::isatty(STDOUT_FILENO))
    throw std::runtime_error("prepare/run须由交互终端启动");
  Fd lock(::open((root / "session.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600));
  if (lock.get() < 0) throw SystemError("session.lock");
  if (::flock(lock.get(), LOCK_EX | LOCK_NB) < 0) throw SystemError("会话锁已占用或不可用");
  CheckProcesses();
  std::array<Json, 3> states;
  for (std::size_t i = 0; i < kChannels.size(); ++i) states[i] = CanInfo(kChannels[i]);
  if (options.action == "prepare") {
    Run(inspect);
    std::vector<std::string> down;
    for (std::size_t i = 0; i < kChannels.size(); ++i) {
      if (Contains(states[i], "flags", "UP") && !Configured(states[i]))
        throw std::runtime_error(std::string(kChannels[i]) + "已UP但参数或状态不符；拒绝改动正在使用的通道");
      if (!Configured(states[i])) down.emplace_back(kChannels[i]);
    }
    if (!down.empty()) {
      std::cout << "准备can4–6，CAN FD 1M/2M one-shot；尚不发送电机目标。\n" << std::flush;
      Run({"sudo", "-v"});
      for (const auto& channel : down) {
        Run({"sudo", "-n", "ip", "link", "set", channel, "type", "can", "bitrate", "1000000",
             "dbitrate", "2000000", "fd", "on", "one-shot", "on"});
        Run({"sudo", "-n", "ip", "link", "set", channel, "up"});
      }
    }
    for (const auto* channel : kChannels)
      if (!Configured(CanInfo(channel))) throw std::runtime_error("CAN配置后状态检查未通过");
    std::cout << "CAN已准备。下一步仍须现场输入START-HOME才会使能。\n" << std::flush;
    return 0;
  }
  for (const auto& state : states)
    if (!Configured(state)) throw std::runtime_error("CAN未准备或状态异常；没有启动SDK");
  fs::create_directories(root / "records");
  const auto output = root / "records" / RecordName();
  if (fs::exists(output)) throw std::runtime_error("记录目录已存在，拒绝覆盖");
  EchoGuard echo;
  const int flags = ::fcntl(lock.get(), F_GETFD);
  if (flags < 0 || ::fcntl(lock.get(), F_SETFD, flags & ~FD_CLOEXEC) < 0)
    throw SystemError("继承会话锁");
  std::vector<std::string> command{
      binary.string(), "--config", config.string(), "--backend", "hardware",
      "--step-deg", options.step, "--test-mode", options.test_mode, "--ui", "--output", output.string()};
  auto argv_exec = Arguments(command);
  ::execv(binary.c_str(), argv_exec.data());
  throw SystemError("启动机械臂控制程序");
}
}  // namespace

int main(int argc, char** argv) {
  try { return Main(argc, argv); }
  catch (const std::exception& error) {
    std::cerr << "拒绝/失败：" << error.what() << '\n';
    return 1;
  }
}
