// Ubuntu-only offline replay; no SDK linkage or device interfaces.
#include <iostream>
#include <iomanip>
#include <memory>
#include "runtime/arm_startup_check.h"
using namespace sckd::io::axischeck;
int main() {
  Joints home; for(auto& v:home) if(!(std::cin>>v)) return 2;
  double now,first=-1,ready_at=-1; int count=0; bool initialized=false;
  Policy p; p.step_rad=10*kDeg; p.max_excursion_rad=12*kDeg;
  Controller c(p); std::unique_ptr<StartupCheck> startup;
  Sample s;
  while(std::cin>>now) {
    for(auto& v:s.q) std::cin>>v;
    for(auto& v:s.dq) std::cin>>v;
    for(auto& v:s.stamp_s) std::cin>>v;
    for(auto& v:s.fault_bits) std::cin>>v;
    for(auto& v:s.mode_state) std::cin>>v;
    if(!std::cin) return 2;
    if(first<0) { first=now; startup=std::make_unique<StartupCheck>(now,home,p); }
    ++count;
    if(!initialized) {
      if(startup->Update(s,now)) {
        if(!c.Initialize(s,now,home,startup->feedback_anchor())) { std::cerr<<c.status().reason; return 3; }
        initialized=true; ready_at=now;
      } else if(startup->fault()) { std::cerr<<startup->reason(); return 4; }
    } else {
      const auto out=c.Tick(s,now);
      if(out.fault || out.q!=home) { std::cerr<<out.reason; return 5; }
    }
  }
  if(!initialized || !c.status().settled) return 6;
  std::cout<<std::setprecision(12)<<"{\"passed\":true,\"rows\":"<<count<<",\"startup_ready_after_s\":"<<ready_at-first<<",\"command_unchanged\":true,\"hardware_started\":false}\n";
}
