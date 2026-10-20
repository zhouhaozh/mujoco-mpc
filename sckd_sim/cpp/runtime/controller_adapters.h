#pragma once
#include "runtime/intercept_modules.h"
namespace sckd::runtime {
void FillBasePred(HitPlan&,SimBackend&,double,double,double);
bool RunOcp(WholebodyOcp&,HitPlan&,SimBackend&,double,WboResult*);
bool RunWbmpc(WholebodyMpc&,const HitPlan&,SimBackend&,ArmIk&,double,const double[5],const double[5],bool,const BaseRefGen::Schedule&,WmPlan*,ControlTrace&);
}
