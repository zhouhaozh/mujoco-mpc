#pragma once
#include "core/types.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace sckd {
// 原生采集帧是控制输入；插值位置/速度只用于外源参考回放和诊断。
class MeasuredBallReplay {
 struct Row { double t; std::array<double,3> p; };
 std::vector<Row> rows_;size_t next_=0;
 public:
 explicit MeasuredBallReplay(const char* path) {
  std::ifstream f(path);if(!f)throw std::runtime_error("cannot open measured replay");
  std::string line;std::getline(f,line);
  if(line!="t_s,x_m,y_m,z_m")throw std::runtime_error("invalid measured replay header");
  while(std::getline(f,line)) {
   std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);Row r;
   if(!(s>>r.t>>r.p[0]>>r.p[1]>>r.p[2]))throw std::runtime_error("invalid measured row");
   std::string extra;if(s>>extra)throw std::runtime_error("extra measured column");
   if(!std::isfinite(r.t)||r.t<0||(!rows_.empty()&&r.t<=rows_.back().t))throw std::runtime_error("invalid measured clock");
   for(double v:r.p)if(!std::isfinite(v))throw std::runtime_error("nonfinite measured position");
   rows_.push_back(r);
  }
  if(rows_.size()<2||rows_.front().t!=0)throw std::runtime_error("measured replay must start at zero with 2+ frames");
 }
 double end() const {return rows_.back().t;}
 size_t received() const {return next_;}
 bool TakeReceived(double now,BallObs* out) {
  if(next_>=rows_.size()||rows_[next_].t>now+1e-12)return false;
  const auto& r=rows_[next_++];out->t=r.t;for(int j=0;j<3;++j)out->p[j]=r.p[j];return true;
 }
 void Reference(double t,double p[3],double v[3]) const {
  auto it=std::upper_bound(rows_.begin(),rows_.end(),t,[](double t,const Row& r){return t<r.t;});
  size_t i=std::clamp<size_t>(it-rows_.begin(),1,rows_.size()-1);
  const auto& a=rows_[i-1];const auto& b=rows_[i];double dt=b.t-a.t;
  double u=std::clamp((t-a.t)/dt,0.,1.);
  for(int j=0;j<3;++j){p[j]=a.p[j]+u*(b.p[j]-a.p[j]);v[j]=(b.p[j]-a.p[j])/dt;}
 }
};
}
