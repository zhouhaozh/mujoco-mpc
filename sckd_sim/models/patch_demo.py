#!/usr/bin/env python3
"""demo 场景手工注入项维护脚本(build_model.py 只产 scene,demo 的球/拍/球场为手注入)。

在 models/generated/x4_tennis_demo.{mjcf,xml} 上应用(幂等):
  1. 球:网球黄绿
  2. 拍:racket_face 碰撞盒隐藏(group 3 + alpha 0,不再遮住拍框);Link5 网格
     (拍柄+拍框)银白;拍面接触参数 solref/friction
  3. 机器人:其余网格 geom 全黑
  4. 标准网球场(docs/球场平面图.png:23.77×10.97,发球线距网 6.40,网 12.8×1.07):
     深绿围区 + 蓝场 + 白线(纯视觉);球网可碰撞(击球后真实触网)
机器人底线在 x=0,网在 x=+11.885,对端底线 x=+23.77;宽度以 y=0 居中。
重新生成模型后重跑本脚本即可。用法: python models/patch_demo.py
"""
import re
from pathlib import Path

GEN = Path(__file__).resolve().parent / "generated"
BALL_RGBA = "0.8 1.0 0.1 1"          # 网球黄绿(optic yellow)
RACKET_RGBA = "0.85 0.87 0.9 1"      # 银白(拍框/柄)
RACKET_HIDDEN = "0.85 0.87 0.9 0"    # 拍面碰撞盒:隐形
ROBOT_RGBA = "0.06 0.06 0.08 1"      # 黑
RACKET_SOLREF = "0.005 1"
RACKET_FRICTION = "0.8 0.005 0.0001"

# 球场配色(球场平面图:深绿围区/蓝场/白线)
C_GREEN = "0.13 0.35 0.16 1"
C_BLUE = "0.16 0.36 0.60 1"
C_ALLEY = "0.12 0.27 0.48 1"
C_LINE = "0.95 0.95 0.95 1"
C_NET = "0.15 0.15 0.15 0.55"

HALF_L = 11.885     # 半场长(底线→网)
W_D = 5.485         # 双打半宽
W_S = 4.115         # 单打半宽
SVC = 6.40          # 发球线→网
LW = 0.05           # 线宽(半宽 0.025)


def court_xml() -> str:
    g = []  # (name, pos, size, rgba) 视觉 geom
    def box(name, x, y, z, hx, hy, hz, rgba):
        g.append(f'<geom name="{name}" type="box" pos="{x} {y} {z}" size="{hx} {hy} {hz}" '
                 f'rgba="{rgba}" contype="0" conaffinity="0" mass="0" />')
    # 围区(深绿):总区长 36.57/宽 18.29,底线在 x=0 → 两侧各延 6.40
    box("ct_surround", HALF_L, 0, 0.001, 18.285, 9.145, 0.0005, C_GREEN)
    # 场地(蓝):x∈[0,23.77]
    box("ct_court", HALF_L, 0, 0.002, HALF_L, W_D, 0.0005, C_BLUE)
    # 双打巷(深蓝,单打线与双打线之间)
    box("ct_alley_l", HALF_L, -(W_S + W_D) / 2, 0.003, HALF_L, (W_D - W_S) / 2, 0.0005, C_ALLEY)
    box("ct_alley_r", HALF_L, (W_S + W_D) / 2, 0.003, HALF_L, (W_D - W_S) / 2, 0.0005, C_ALLEY)
    # 白线(z 0.004)
    z = 0.004
    box("ct_base_near", LW / 2, 0, z, LW / 2, W_D, 0.0005, C_LINE)              # 本方底线
    box("ct_base_far", 2 * HALF_L - LW / 2, 0, z, LW / 2, W_D, 0.0005, C_LINE)  # 对端底线
    box("ct_side_d_l", HALF_L, -W_D + LW / 2, z, HALF_L, LW / 2, 0.0005, C_LINE)
    box("ct_side_d_r", HALF_L, W_D - LW / 2, z, HALF_L, LW / 2, 0.0005, C_LINE)
    box("ct_side_s_l", HALF_L, -W_S, z, HALF_L, LW / 2, 0.0005, C_LINE)
    box("ct_side_s_r", HALF_L, W_S, z, HALF_L, LW / 2, 0.0005, C_LINE)
    box("ct_svc_near", HALF_L - SVC, 0, z, LW / 2, W_S, 0.0005, C_LINE)         # 发球线×2
    box("ct_svc_far", HALF_L + SVC, 0, z, LW / 2, W_S, 0.0005, C_LINE)
    box("ct_svc_mid", HALF_L, 0, z, SVC, LW / 2, 0.0005, C_LINE)                # 发球区中线
    box("ct_mark_near", 0.15, 0, z, 0.10, LW / 2, 0.0005, C_LINE)               # 底线中点标记
    box("ct_mark_far", 2 * HALF_L - 0.15, 0, z, 0.10, LW / 2, 0.0005, C_LINE)
    net = (f'<geom name="ct_net" type="box" pos="{HALF_L} 0 0.475" size="0.012 6.4 0.475" '
           f'rgba="{C_NET}" contype="1" conaffinity="1" solref="0.02 1" mass="0" />'
           f'<geom name="ct_post_l" type="cylinder" pos="{HALF_L} -6.4 0.535" size="0.04 0.535" '
           f'rgba="0.3 0.3 0.32 1" contype="1" conaffinity="1" mass="0" />'
           f'<geom name="ct_post_r" type="cylinder" pos="{HALF_L} 6.4 0.535" size="0.04 0.535" '
           f'rgba="0.3 0.3 0.32 1" contype="1" conaffinity="1" mass="0" />')
    return ('<body name="court" pos="0 0 0">' + "".join(g) + net + "</body>")


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
    # 4) 球场(幂等:先移除旧 court body 再注入,插在 worldbody 结束前)
    s = re.sub(r"<body name=\"court\"[^>]*>.*?</body>", "", s, flags=re.S)
    court = court_xml()
    if "</worldbody>" in s:
        s = s.replace("</worldbody>", court + "</worldbody>", 1)
    else:
        s = s.replace("</mujoco>", "<worldbody>" + court + "</worldbody></mujoco>", 1)
    # 5) 拍-球接触改解析冲量(§3.5 已知碰撞模型,2026-10-07):软接触在合速
    #    >10 m/s 处于穿透态(停球距离 v·τ ≫ 面厚+球半径的几何预算,1ms 步长下
    #    timeconst 收到稳定界内也救不回,受控台架实测 R²<0)——排除求解器
    #    接触对,由 app 真值侧 cpp/sim/analytic_contact.h 施加参数化冲量;
    #    planner ReboundInverse 用同一映射逆解(自洽),实机实测后换 k
    s = re.sub(r'<contact>\s*<exclude name="ball_vs_racket"[^>]*/>\s*</contact>', "", s)
    s = re.sub(r'<exclude name="ball_vs_racket"[^>]*/>', "", s)
    s = s.replace("</worldbody>",
                  '</worldbody><contact><exclude name="ball_vs_racket" '
                  'body1="ball" body2="Link5"/></contact>', 1)
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
