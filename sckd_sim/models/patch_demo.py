#!/usr/bin/env python3
"""demo 场景手工注入项维护脚本(build_model.py 只产 scene,demo 的球/拍为手注入)。

在 models/generated/x4_tennis_demo.{mjcf,xml} 上应用:
  1. 球:网球黄绿(视觉识别配色)
  2. 拍面 racket_face:银白 + 接触参数(真实拍-球触 ~5ms,默认 solref=0.02 过软)
  3. 机器人:网格 geom 全部黑色
重新生成模型后重跑本脚本即可。
用法: python models/patch_demo.py
"""
import re
from pathlib import Path

GEN = Path(__file__).resolve().parent / "generated"
BALL_RGBA = "0.8 1.0 0.1 1"          # 网球黄绿(optic yellow)
RACKET_RGBA = "0.85 0.87 0.9 1"      # 银白
ROBOT_RGBA = "0.06 0.06 0.08 1"      # 黑
RACKET_SOLREF = "0.005 1"
RACKET_FRICTION = "0.8 0.005 0.0001"

for name in ("x4_tennis_demo.mjcf", "x4_tennis_demo.xml"):
    p = GEN / name
    s = p.read_text()
    # 1) 球配色
    s = re.sub(r'(<geom name="ball_geom"[^>]*rgba=")[^"]*(")', r"\g<1>" + BALL_RGBA + r"\g<2>", s)
    # 2) 拍面:配色 + 接触参数(幂等:先剥离旧值再注入)
    s = re.sub(r'(<geom name="racket_face"[^>]*?)\s*(solref="[^"]*")', r"\g<1>", s)
    s = re.sub(r'(<geom name="racket_face"[^>]*?)\s*(friction="[^"]*")', r"\g<1>", s)
    s = re.sub(r'(<geom name="racket_face"[^>]*rgba=")[^"]*(")', r"\g<1>" + RACKET_RGBA + r"\g<2>", s)
    s = s.replace('<geom name="racket_face"',
                  f'<geom name="racket_face" solref="{RACKET_SOLREF}" friction="{RACKET_FRICTION}"', 1)
    # 3) 机器人网格 geom 全黑(不动球/地/拍面/已隐藏的碰撞代理)
    def black(m):
        g = m.group(0)
        if ('name="ball_geom"' in g or 'name="floor"' in g
                or 'name="racket_face"' in g or 'rgba="0.5 0.5 0.5 0"' in g):
            return g
        return re.sub(r'rgba="[^"]*"', f'rgba="{ROBOT_RGBA}"', g)
    s = re.sub(r"<geom [^>]*mesh=\"[^\"]+\"[^>]*/>", black, s)
    p.write_text(s)
    print(f"[OK] {p}")
print("done")
