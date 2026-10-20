#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {

void AdvanceSimulation(PhysicsContext c, double ts) {
  auto& be = c.be;
  auto& measured = c.measured;
  auto& set_measured_reference = c.set_measured_reference;
  auto& m = c.m;
  auto& d = c.d;
  auto& ric_pp = c.ric_pp;
  auto& gnd_pp = c.gnd_pp;
  auto& calib_saved = c.calib_saved;
  auto& video_ground_contacts = c.video_ground_contacts;
  auto& ball_qadr = c.ball_qadr;
  auto& ball_vadr = c.ball_vadr;
  auto& face_gid = c.face_gid;
  auto& site_id = c.site_id;
  auto& t_ground = c.t_ground;
  auto& calib_vin = c.calib_vin;
  auto& calib_win = c.calib_win;
  auto& calib_n = c.calib_n;
  auto& calib_bp = c.calib_bp;
  auto& calib_sp = c.calib_sp;
  auto& calib_rv = c.calib_rv;
  auto& calib_other = c.calib_other;
  // 拍-球解析冲量(§3.5 已知碰撞模型;求解器接触对已排除,真值侧在此施
  // 加)。命中事件接管 [calib] 标定捕获
  if(!measured) {
    const RacketImpulseEvent ev =
        ApplyRacketImpulse(m, d, face_gid, ball_qadr, ball_vadr, ric_pp);
    if (ev.fired) {
      mj_kinematics(m, d);
      double rp[3] = {0, 0, 0};
      mju_rotVecQuat(rp, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
      const double* sbp = d->xpos + m->site_bodyid[site_id]*3;
      for (int k = 0; k < 3; ++k) {
        calib_vin[k] = ev.v_in[k]; calib_win[k] = d->qvel[ball_vadr+3+k];
        calib_n[k] = -ev.n[k];      // [calib] 口径:n=球→拍(内向)
        calib_bp[k] = d->qpos[ball_qadr+k];
        calib_sp[k] = sbp[k] + rp[k]; calib_rv[k] = ev.v_face[k];
      }
      std::snprintf(calib_other, sizeof(calib_other), "racket_face");
      calib_saved = true;
      std::printf("  [touch-a] v_close=%.2f → vn_out=%.2f off=(%.3f,%.3f) n=(%.2f,%.2f,%.2f) vout=(%.1f,%.1f,%.1f)\n",
                  ev.v_close, ev.vn_out, ev.off_l, ev.off_h,
                  ev.n[0], ev.n[1], ev.n[2],
                  d->qvel[ball_vadr], d->qvel[ball_vadr+1], d->qvel[ball_vadr+2]);
    }
  }
  // 地面解析冲量(§3.5 地面支;模型已排除 ball↔floor 求解器对,不调则
  // 球穿地)。事件置 t_ground:保持"落地后撞拍无效"判据语义(解析冲量
  // 下球不会持续处于 z<R 且 vz<0,旧观测式永不触发)
  if(!measured) {
    const GroundImpulseEvent gev =
        ApplyGroundImpulse(d, ball_qadr, ball_vadr, gnd_pp);
    if (gev.fired && !gev.resting) {
      ++video_ground_contacts;
      if (t_ground < 0) t_ground = ts;
    }
  }
  StageProfile::Measure("physics_and_motor_step", [&]{be.run(1);});
  if(measured)set_measured_reference(ts);


}
}
