#include "estimation/measured_ball_replay.h"
#define CHECK(x) do { if(!(x)) throw std::runtime_error(#x); } while(false)
#include <fstream>
#include <iostream>
#include <cstdio>
int main(int argc,char**argv) {
 if(argc!=2)return 2;std::string prefix=argv[1];
 auto save=[&](const std::string& path,const std::string& rows){std::ofstream f(path);f<<"t_s,x_m,y_m,z_m\n"<<rows;};
 const std::string a=prefix+"-a.csv",b=prefix+"-b.csv",bad=prefix+"-bad.csv";
 save(a,"0,1,2,3\n0.005,2,3,4\n0.01,3,4,5\n");
 save(b,"0,1,2,3\n0.005,2,3,4\n0.01,3,4,5\n0.02,999,999,999\n");
 sckd::MeasuredBallReplay x(a.c_str()),y(b.c_str());sckd::BallObs u,v;
 for(double now:{0.,.004,.005,.006,.01}) {
  while(true){bool ax=x.TakeReceived(now,&u),by=y.TakeReceived(now,&v);CHECK(ax==by);if(!ax)break;
   CHECK(u.t<=now&&u.t==v.t);for(int j=0;j<3;++j)CHECK(u.p[j]==v.p[j]);}
 }
 CHECK(x.received()==3&&y.received()==3);CHECK(!y.TakeReceived(.019,&v));CHECK(y.TakeReceived(.02,&v)&&v.p[0]==999);
 double p[3],speed[3];x.Reference(.0025,p,speed);CHECK(p[0]==1.5&&speed[0]==200);
 CHECK(x.received()==3); // reference lookup cannot advance observation clock
 save(bad,"0,1,2,3\n0,1,2,3\n");bool rejected=false;try{sckd::MeasuredBallReplay r(bad.c_str());}catch(...){rejected=true;}CHECK(rejected);
 std::remove(a.c_str());std::remove(b.c_str());std::remove(bad.c_str());
 std::cout<<"[PASS] future-prefix invariance; native timestamps; reference isolated; duplicate rejected\n";
}
