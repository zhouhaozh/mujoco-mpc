#pragma once
#include "control/legacy_nmpc.h"
namespace sckd::runtime {
// 此层只作现行守门决策；保留各算法计数差异，不新增超时/过期政策。
class PlanSupervisor {
 public:
  static bool CanApplyNmpc(const LegacyNmpcResult& r,bool coherent,double remaining,double strict_window_s=.100){return coherent||remaining>=strict_window_s||(r.status==LegacyNmpcStatus::kOk&&r.solver_ok&&r.solve_ms<10.0);}
  static bool NmpcFallback(bool have,int& failures,bool stale){return (!have&&++failures>15)||stale;}
  static bool WbmpcFallback(bool have,int& failures){return ++failures>(have?20:15);}
};
}
