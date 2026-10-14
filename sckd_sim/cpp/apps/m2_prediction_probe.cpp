#include "estimation/socket_ball_predictor.h"
#include <iostream>
#include <sstream>
#include <algorithm>
int main(int argc,char** argv) {
 try {
  if(argc!=2)throw std::runtime_error("usage: m2_prediction_probe unix_socket < t,x,y,z lines");
  sckd::SocketBallPredictor predictor(argv[1]);std::string line;
  while(std::getline(std::cin,line)) {
   std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);sckd::BallObs o;
   if(!(in>>o.t>>o.p[0]>>o.p[1]>>o.p[2]))throw std::runtime_error("invalid observation row");
   predictor.Update(o);
  }
  nlohmann::json out={{"initialized",predictor.initialized()},{"t_s",predictor.t_last()}};
  out["samples"]=nlohmann::json::array();
  if(predictor.initialized())for(double h:{.1,.2,.5,.7,.9,1.2,1.5,1.6}) {
   double p[3],v[3];bool ok=predictor.PredictFull(predictor.t_last()+h,p,v);
   nlohmann::json q={{"horizon_s",h},{"valid",ok}};if(ok){q["p"]={p[0],p[1],p[2]};q["v"]={v[0],v[1],v[2]};}out["samples"].push_back(q);
  }
  std::cout<<out.dump()<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
