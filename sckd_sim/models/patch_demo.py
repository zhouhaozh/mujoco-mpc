#!/usr/bin/env python3
"""demo 场景手工注入项维护脚本(build_model.py 只产 scene,demo 的球/拍/球场为手注入)。

在 models/generated/x4_tennis_demo.{mjcf,xml} 上应用(幂等):
  1. 球:网球黄绿
  2. 拍:racket_face 碰撞盒隐藏(group 3 + alpha 0,不再遮住拍框);Link5 网格
     (拍柄+拍框)银白;拍面接触参数 solref/friction
  3. 机器人:其余网格 geom 全黑
  4. 标准羽毛球场：13.40×6.10 m，单打宽5.18 m，40 mm白线；球网可碰撞。
机器人底线在 x=0,网在 x=+6.70,对端底线 x=+13.40;宽度以 y=0 居中。
本脚本仅替换场地；原演示的网球飞行/回弹/发球逻辑仍为旧物理模型。
重新生成模型后重跑本脚本即可。用法: python models/patch_demo.py
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from badminton_court import add_badminton_court

GEN = Path(__file__).resolve().parent / "generated"
BALL_RGBA = "0.8 1.0 0.1 1"          # 网球黄绿(optic yellow)
RACKET_RGBA = "0.85 0.87 0.9 1"      # 银白(拍框/柄)
RACKET_HIDDEN = "0.85 0.87 0.9 0"    # 拍面碰撞盒:隐形
ROBOT_RGBA = "0.06 0.06 0.08 1"      # 黑
RACKET_SOLREF = "0.005 1"
RACKET_FRICTION = "0.8 0.005 0.0001"

for name in ("x4_tennis_demo.mjcf", "x4_tennis_demo.xml"):
    p = GEN / name
    s = p.read_text()
    # 1) 球配色
    s = re.sub(r'(<geom name="ball_geom"[^>]*rgba=")[^"]*(")', r"\g<1>" + BALL_RGBA + r"\g<2>", s)
    # 2) 拍面碰撞盒:隐形 + group 3;接触参数(先剥离旧值再注入,幂等)
    s = re.sub(r'(<geom name="racket_face"[^>]*?)\s*(solref="[^"]*")', r"\g<1>", s)
    s = re.sub(r'(<geom name="racket_face"[^>]*?)\s*(friction="[^"]*")', r"\g<1>", s)
    s = re.sub(r'(<geom name="racket_face"[^>]*?)\s*(group="[^"]*")', r"\g<1>", s)
    s = re.sub(r'(<geom name="racket_face"[^>]*rgba=")[^"]*(")', r"\g<1>" + RACKET_HIDDEN + r"\g<2>", s)
    s = s.replace('<geom name="racket_face"',
                  f'<geom name="racket_face" group="3" solref="{RACKET_SOLREF}" friction="{RACKET_FRICTION}"', 1)
    # 3) 机器人网格 geom:Link5(拍柄+框)银白,其余黑
    def recolor(m):
        g = m.group(0)
        if ('name="ball_geom"' in g or 'name="floor"' in g
                or 'name="racket_face"' in g or 'rgba="0.5 0.5 0.5 0"' in g
                or 'mesh="ct_' in g):
            return g
        color = RACKET_RGBA if 'mesh="Link5"' in g else ROBOT_RGBA
        return re.sub(r'rgba="[^"]*"', f'rgba="{color}"', g)
    s = re.sub(r"<geom [^>]*mesh=\"[^\"]+\"[^>]*/>", recolor, s)
    # 4) 共用标准羽毛球场；重新运行会替换同名 body 和网格资产。
    root = ET.fromstring(s)
    add_badminton_court(root, collision_enabled=True)
    s = ET.tostring(root, encoding="unicode")
    # 5) 拍-球接触改解析冲量(§3.5 已知碰撞模型,2026-10-07):软接触在合速
    #    >10 m/s 处于穿透态(停球距离 v·τ ≫ 面厚+球半径的几何预算,1ms 步长下
    #    timeconst 收到稳定界内也救不回,受控台架实测 R²<0)——排除求解器
    #    接触对,由 app 真值侧 cpp/sim/analytic_contact.h 施加参数化冲量;
    #    planner ReboundInverse 用同一映射逆解(自洽),实机实测后换 k
    # 6) 地面接触改解析冲量(§3.5 地面支,2026-10-08):原始求解器 ball↔floor
    #    为 solref(0.02,1) 临界阻尼,无旋落球 e_z≈0.09≈死球(ground_calib.py
    #    实测,与网球 ITF 0.73 差 8 倍)→ 排除 ball↔floor(worldbody 直接可
    #    碰撞 geom 仅 floor 一个,已核实;球网/立柱在 court body 不受影响),
    #    真值侧 analytic_contact.h ApplyGroundImpulse(e_z=0.73 文献初值/
    #    μ=0.20 台架无旋拟合,预测器 BounceParams 同参数自洽;平抽 v1 无旋,
    #    旋转耦合后续迭代)。app 必须每拍调 ApplyGroundImpulse,否则球穿地。
    s = re.sub(r'<contact>\s*<exclude name="ball_vs_racket"[^>]*/>\s*</contact>', "", s)
    s = re.sub(r'<exclude name="ball_vs_racket"[^>]*/>', "", s)
    s = re.sub(r'\s*<exclude name="ball_vs_floor"[^>]*/>', "", s)
    s = re.sub(r'<contact>\s*</contact>|<contact\s*/>', "", s)
    s = s.replace("</worldbody>",
                  '</worldbody><contact>'
                  '<exclude name="ball_vs_racket" body1="ball" body2="Link5"/>'
                  '<exclude name="ball_vs_floor" body1="ball" body2="world"/>'
                  '</contact>', 1)
    # 注:待机位姿(实机 sckd-robot home,J2 抬臂 35°)不写入 keyframe —— keyframe
    # 零位是 M0/M1 验收基线;且 app 侧 IK 种子取实测 qpos,keyframe 抬臂会改变
    # 解支(实测命中 5/10→2/10)。待机由 m2_demo 自主驱动(待机→就位 0.25s 过渡)。
    m = re.search(r'<key name="home" qpos="([^"]*)"', s)
    if m:
        vals = m.group(1).split()
        if len(vals) >= 24:
            vals[19:24] = ["0.0"] * 5  # 主动归零(清历史残留)
            s = s[:m.start(1)] + " ".join(vals) + s[m.end(1):]
    p.write_text(s)
    print(f"[OK] {p}")
print("done")
