#pragma once
// Opt-in research profiling. Samples stay in memory and flush once on exit.
// Inclusive times may be nested; their sums must not be added together.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
namespace sckd {
class StageProfile {
 using Clock=std::chrono::steady_clock;
 std::string path_; Clock::time_point start_=Clock::now();
 std::map<std::string,std::vector<double>> values_;
 std::map<std::string,long long> counts_;
 public:
 StageProfile() {if(const char* p=std::getenv("M2_TIMING_PATH"))path_=p;}
 static StageProfile& Get() {static StageProfile p;return p;}
 bool enabled() const {return !path_.empty();}
 void Add(const std::string& key,double ms) {values_[key].push_back(ms);}
 void Count(const char* key) {if(enabled())++counts_[key];}
 class Scope {
  const char* key_; Clock::time_point start_;
  public:
  explicit Scope(const char* key):key_(Get().enabled()?key:nullptr) {if(key_)start_=Clock::now();}
  ~Scope() {if(key_)Get().Add(key_,std::chrono::duration<double,std::milli>(Clock::now()-start_).count());}
 };
 template<class F> static decltype(auto) Measure(const char* key,F&& f) {Scope s(key);return f();}
 ~StageProfile() {
  if(!enabled())return;
  nlohmann::json result={{"schema","STAGE_TIMING/1"},{"process_wall_s",std::chrono::duration<double>(Clock::now()-start_).count()},{"events",counts_}};
  for(auto& kv:values_) {
   auto& a=kv.second;double sum=std::accumulate(a.begin(),a.end(),0.);double first=a.front();std::sort(a.begin(),a.end());
   auto quant=[&](double q){return a[std::min(a.size()-1,size_t(std::ceil(q*a.size())-1))];};
   result["stages"][kv.first]={{"n",a.size()},{"total_ms",sum},{"mean_ms",sum/a.size()},{"p95_ms",quant(.95)},{"p99_ms",quant(.99)},{"max_ms",a.back()},{"first_ms",first},{"over_1ms",std::count_if(a.begin(),a.end(),[](double v){return v>1;})},{"over_2ms",std::count_if(a.begin(),a.end(),[](double v){return v>2;})},{"over_20ms",std::count_if(a.begin(),a.end(),[](double v){return v>20;})}};
  }
  std::ofstream f(path_);f<<result.dump(2)<<'\n';
 }
};
}
