#include "estimation/measured_shuttle_predictor.h"

#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Json = nlohmann::json;
namespace fs = std::filesystem;

std::runtime_error SystemError(const std::string& operation) {
  return std::runtime_error(operation + ": " + std::strerror(errno));
}

std::string AbsolutePath(const Json& document, const char* key) {
  const fs::path path = document.at(key).get<std::string>();
  if (!path.is_absolute()) throw std::runtime_error(std::string(key) + " must be absolute");
  return path.string();
}

sckd::MeasuredShuttlePredictor::Config LoadConfig(const fs::path& config_path) {
  if (!config_path.is_absolute()) throw std::runtime_error("config path must be absolute");
  std::ifstream input(config_path);
  if (!input) throw std::runtime_error("cannot open config: " + config_path.string());
  Json document;
  input >> document;
  if (!document.is_object()) throw std::runtime_error("predictor config must be a JSON object");

  fs::path project_root = document.at("project_root").get<std::string>();
  if (project_root.is_relative()) project_root = config_path.parent_path() / project_root;
  if (!fs::is_directory(project_root))
    throw std::runtime_error("project_root is not a directory: " + project_root.string());

  sckd::MeasuredShuttlePredictor::Config config;
  config.trajectory_path = AbsolutePath(document, "trajectory_path");
  config.net20_model_path = AbsolutePath(document, "ball_mlp_model_path");
  config.net20_normalizer_path = AbsolutePath(document, "ball_mlp_normalizer_path");
  config.fixed_height_m = document.value("fixed_height_m", config.fixed_height_m);
  config.height_tolerance_m =
      document.value("height_tolerance_m", config.height_tolerance_m);
  config.initial_estimated_hit_s =
      document.value("initial_estimated_hit_s", config.initial_estimated_hit_s);
  config.preassociated_replay = document.value("preassociated_replay", false);
  return config;
}

class ListeningSocket {
 public:
  explicit ListeningSocket(const std::string& path) : path_(path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof(address.sun_path))
      throw std::runtime_error("Unix socket path is empty or too long");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);

    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) throw SystemError("socket");
    const socklen_t address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
    if (bind(fd_, reinterpret_cast<const sockaddr*>(&address), address_length) != 0) {
      const auto error = SystemError("bind " + path);
      close(fd_);
      fd_ = -1;
      throw error;
    }
    owns_path_ = true;
    if (listen(fd_, 1) != 0) {
      const auto error = SystemError("listen");
      close(fd_);
      fd_ = -1;
      unlink(path_.c_str());
      owns_path_ = false;
      throw error;
    }
  }

  ~ListeningSocket() {
    if (fd_ >= 0) close(fd_);
    if (owns_path_) unlink(path_.c_str());
  }

  ListeningSocket(const ListeningSocket&) = delete;
  ListeningSocket& operator=(const ListeningSocket&) = delete;

  int Accept() const {
    const int client = accept(fd_, nullptr, nullptr);
    if (client < 0) throw SystemError("accept");
    return client;
  }

 private:
  std::string path_;
  int fd_ = -1;
  bool owns_path_ = false;
};

void SendLine(int fd, const Json& response) {
  const std::string line = response.dump() + '\n';
  std::size_t offset = 0;
  while (offset < line.size()) {
    const ssize_t sent = send(fd, line.data() + offset, line.size() - offset, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) continue;
    if (sent <= 0) throw SystemError("send");
    offset += static_cast<std::size_t>(sent);
  }
}

Json TargetResponse(const sckd::MeasuredShuttlePredictor& predictor) {
  const auto& target = predictor.target();
  const auto& stats = predictor.stats();
  return {{"ok", true},
          {"target",
           {{"valid", target.valid},
            {"p_hit_m", target.position_m},
            {"v_in_m_s", target.incoming_velocity_m_s},
            {"t_hit_s", target.hit_time_s},
            {"prediction_ms", target.prediction_ms}}},
          {"predicted_trajectory", predictor.predicted_trajectory()},
          {"stats",
           {{"received", stats.received},
            {"accepted", stats.accepted},
            {"rejected", stats.rejected},
            {"input_policy", stats.input_policy}}},
          {"end_time_s", predictor.end_time_s()}};
}

bool HandleLine(int client, const std::string& line,
                sckd::MeasuredShuttlePredictor& predictor) {
  try {
    const Json request = Json::parse(line);
    if (!request.is_object()) throw std::runtime_error("request must be a JSON object");
    if (request.value("op", std::string()) == "close") {
      SendLine(client, {{"ok", true}});
      return false;
    }
    predictor.UpdateUntil(request.at("t").get<double>(),
                          request.value("refresh_prediction", true));
    SendLine(client, TargetResponse(predictor));
  } catch (const std::exception& error) {
    SendLine(client, {{"ok", false}, {"error", error.what()}});
  }
  return true;
}

void ServeClient(int client, sckd::MeasuredShuttlePredictor& predictor) {
  std::string buffered;
  char chunk[4096];
  while (true) {
    const ssize_t received = recv(client, chunk, sizeof(chunk), 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) return;
    buffered.append(chunk, static_cast<std::size_t>(received));
    if (buffered.size() > 1024 * 1024) {
      SendLine(client, {{"ok", false}, {"error", "request line too long"}});
      return;
    }
    std::size_t newline = 0;
    while ((newline = buffered.find('\n')) != std::string::npos) {
      const std::string line = buffered.substr(0, newline);
      buffered.erase(0, newline + 1);
      if (!line.empty() && !HandleLine(client, line, predictor)) return;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: x4_shuttle_predictor SOCKET_PATH ABSOLUTE_CONFIG_PATH\n";
    return 2;
  }
  try {
    sckd::MeasuredShuttlePredictor predictor(LoadConfig(argv[2]));
    ListeningSocket server(argv[1]);
    std::cout << "x4_shuttle_predictor listening on " << argv[1] << std::endl;
    const int client = server.Accept();
    try {
      ServeClient(client, predictor);
      close(client);
    } catch (...) {
      close(client);
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << "x4_shuttle_predictor: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
