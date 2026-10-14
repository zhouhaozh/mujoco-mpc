#pragma once
#include "core/stage_profile.h"
#include "estimation/ball_prediction_interface.h"
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace sckd {
class SocketBallPredictor final : public BallPredictionInterface {
 using Json=nlohmann::json;
 struct Point { double t; std::array<double,3> p,v; bool bounced; };
 int fd_=-1; mutable std::string buffer_; unsigned seq_=0;
 bool init_=false; double last_=-1e30; double x_[6]={};
 mutable bool dirty_=true; mutable double bounce_=-1;
 mutable std::vector<Point> grid_;
 Json Call(Json request) const {
  const auto op=request.value("op",std::string());
  StageProfile::Scope profile(op=="forecast"?"ipc_forecast":op=="update"?"ipc_update":"ipc_reset");
  request["schema"]="BALL_FORECAST/1";
  std::string text=request.dump()+"\n";size_t offset=0;
  while(offset<text.size()) { auto n=send(fd_,text.data()+offset,text.size()-offset,MSG_NOSIGNAL);if(n<=0)throw std::runtime_error("predictor IPC send failed");offset+=n; }
  while(buffer_.find('\n')==std::string::npos) { char b[8192];auto n=recv(fd_,b,sizeof(b),0);if(n<=0)throw std::runtime_error("predictor IPC timeout/disconnect");buffer_.append(b,n);if(buffer_.size()>2000000)throw std::runtime_error("predictor oversized reply"); }
  auto end=buffer_.find('\n');auto r=Json::parse(buffer_.substr(0,end));buffer_.erase(0,end+1);
  if(r.value("schema","")!="BALL_FORECAST/1" || !r.value("ok",false))throw std::runtime_error("predictor rejected request: "+r.value("error",std::string("invalid schema")));
  return r;
 }
 void Refresh() const {
  if(!dirty_ || !init_)return;
  auto r=Call({{"op","forecast"},{"seq",seq_}});
  if(r.at("seq").get<unsigned>()!=seq_)throw std::runtime_error("prediction version mismatch");
  grid_.clear();bounce_=r.at("next_bounce_t_s").get<double>();
  for(auto& a:r.at("grid")) { Point p{a.at(0),{a.at(1),a.at(2),a.at(3)},{a.at(4),a.at(5),a.at(6)},a.at(7)};
   if(!std::isfinite(p.t)||(!grid_.empty()&&p.t<=grid_.back().t))throw std::runtime_error("invalid forecast clock");
   for(int j=0;j<3;++j)if(!std::isfinite(p.p[j])||!std::isfinite(p.v[j]))throw std::runtime_error("nonfinite forecast");
   grid_.push_back(p);
  }
  dirty_=false;
 }
 public:
 explicit SocketBallPredictor(const char* path) {
  sockaddr_un addr{};addr.sun_family=AF_UNIX;
  if(std::strlen(path)>=sizeof(addr.sun_path))throw std::runtime_error("socket path too long");
  std::strcpy(addr.sun_path,path);fd_=socket(AF_UNIX,SOCK_STREAM,0);
  timeval timeout{2,0};setsockopt(fd_,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(fd_,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
  if(connect(fd_,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))) {close(fd_);fd_=-1;throw std::runtime_error("cannot connect prediction service");}
  try { Call({{"op","reset"}}); } catch(...) {close(fd_);fd_=-1;throw;}
 }
 ~SocketBallPredictor() override {if(fd_>=0)close(fd_);}
 SocketBallPredictor(const SocketBallPredictor&)=delete;
 void Update(const BallObs& o) override {
  if(!std::isfinite(o.t))throw std::runtime_error("invalid observation timestamp");
  if(o.t<=last_)return; // latest sample may be repeated by SimBackend each tick
  auto r=Call({{"op","update"},{"seq",++seq_},{"t_s",o.t},{"xyz",{o.p[0],o.p[1],o.p[2]}}});
  init_=r.at("initialized");last_=o.t;dirty_=true;
  if(init_)for(int i=0;i<6;++i)x_[i]=r.at("state").at(i);
 }
 bool initialized() const override {return init_;}
 double t_last() const override {return last_;}
 const double* state() const override {return x_;}
 double NextBounceTime() const override {Refresh();return bounce_;}
 bool PredictFull(double t,double p[3],double v[3]) const override {
  if(!init_||!std::isfinite(t))return false;Refresh();
  if(grid_.empty()||t<grid_.front().t||t>grid_.back().t)return false;
  for(size_t i=0;i<grid_.size();++i) {
   const auto& b=grid_[i];
   if(std::abs(t-b.t)<1e-10) {for(int j=0;j<3;++j){p[j]=b.p[j];v[j]=b.v[j];}return true;}
   if(b.t>t && i) {const auto& a=grid_[i-1];if(a.bounced!=b.bounced)return false;
    double dt=b.t-a.t,s=(t-a.t)/dt;
    for(int j=0;j<3;++j) {
     p[j]=(2*s*s*s-3*s*s+1)*a.p[j]+(s*s*s-2*s*s+s)*dt*a.v[j]+(-2*s*s*s+3*s*s)*b.p[j]+(s*s*s-s*s)*dt*b.v[j];
     v[j]=(6*s*s-6*s)/dt*a.p[j]+(3*s*s-4*s+1)*a.v[j]+(-6*s*s+6*s)/dt*b.p[j]+(3*s*s-2*s)*b.v[j];
    }return true;
   }
  }return false;
 }
};
}
