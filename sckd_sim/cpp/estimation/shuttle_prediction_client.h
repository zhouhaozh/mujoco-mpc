#pragma once

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace sckd {

// Synchronous client for the isolated ball-mlp prediction service. The service
// owns its OpenCV/runtime dependencies; the simulator exchanges newline JSON only.
class ShuttlePredictionClient {
 public:
  explicit ShuttlePredictionClient(const std::string& socket_path) {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (socket_path.empty() || socket_path.size() >= sizeof(address.sun_path)) {
      throw std::invalid_argument("invalid shuttle prediction socket path");
    }
    std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);

    const int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) throw std::runtime_error(SystemError("cannot create prediction socket"));
    timeval timeout{10, 0};
    if (setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
      const std::string error = SystemError("cannot configure prediction socket timeout");
      close(socket_fd);
      throw std::runtime_error(error);
    }
    const socklen_t address_size =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + socket_path.size() + 1);
    if (connect(socket_fd, reinterpret_cast<const sockaddr*>(&address), address_size) != 0) {
      const std::string error = SystemError("cannot connect shuttle prediction service");
      close(socket_fd);
      throw std::runtime_error(error);
    }
    fd_ = socket_fd;
  }

  ~ShuttlePredictionClient() { Close(); }

  ShuttlePredictionClient(const ShuttlePredictionClient&) = delete;
  ShuttlePredictionClient& operator=(const ShuttlePredictionClient&) = delete;

  nlohmann::json Update(double t, bool refresh_prediction = true) {
    if (!std::isfinite(t)) throw std::invalid_argument("prediction update time must be finite");
    return Request({{"t", t}, {"refresh_prediction", refresh_prediction}});
  }

  void Close() noexcept {
    if (fd_ < 0) return;
    try {
      (void)Request({{"op", "close"}});
    } catch (...) {
      // Closing is best effort; the file descriptor still must be released.
    }
    close(fd_);
    fd_ = -1;
    buffer_.clear();
  }

 private:
  static constexpr std::size_t kMaxResponseBytes = 1024 * 1024;

  static std::string SystemError(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
  }

  nlohmann::json Request(const nlohmann::json& request) {
    if (fd_ < 0) throw std::runtime_error("shuttle prediction client is closed");
    const std::string payload = request.dump() + "\n";
    std::size_t offset = 0;
    while (offset < payload.size()) {
      const ssize_t sent = send(fd_, payload.data() + offset, payload.size() - offset, MSG_NOSIGNAL);
      if (sent < 0 && errno == EINTR) continue;
      if (sent <= 0) throw std::runtime_error(SystemError("prediction socket write failed"));
      offset += static_cast<std::size_t>(sent);
    }

    std::size_t newline = buffer_.find('\n');
    while (newline == std::string::npos) {
      char chunk[4096];
      const ssize_t received = recv(fd_, chunk, sizeof(chunk), 0);
      if (received < 0 && errno == EINTR) continue;
      if (received < 0) throw std::runtime_error(SystemError("prediction socket read failed"));
      if (received == 0) throw std::runtime_error("prediction service closed the socket");
      buffer_.append(chunk, static_cast<std::size_t>(received));
      if (buffer_.size() > kMaxResponseBytes) {
        throw std::runtime_error("prediction response exceeds 1 MiB");
      }
      newline = buffer_.find('\n');
    }

    const std::string line = buffer_.substr(0, newline);
    buffer_.erase(0, newline + 1);
    nlohmann::json response;
    try {
      response = nlohmann::json::parse(line);
    } catch (const nlohmann::json::exception& error) {
      throw std::runtime_error(std::string("invalid prediction JSON response: ") + error.what());
    }
    if (!response.contains("ok") || !response["ok"].is_boolean()) {
      throw std::runtime_error("prediction response is missing boolean ok");
    }
    if (!response["ok"].get<bool>()) {
      const std::string error = response.contains("error") && response["error"].is_string()
                                    ? response["error"].get<std::string>()
                                    : "prediction service rejected request";
      throw std::runtime_error(error);
    }
    return response;
  }

  int fd_ = -1;
  std::string buffer_;
};

}  // namespace sckd
