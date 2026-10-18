#!/usr/bin/env python3
# 底盘阶跃速度响应绘图+metrics(2026-10-16):读 m1_step_resp CSV,绘
# ①vx 阶跃族跟踪(原始指令/斜坡指令/实测+误差) ②vy 阶跃(舵转 90°动态)
# ③wz 阶跃;打印 延迟/上升10-90%/超调/进入±2%带/稳态误差。
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def metrics(t, y, t0, target, pre, span=1.0):
    """阶跃指标:延迟(10%交越)/上升(10→90%)/超调/±2%入带/稳态误差。"""
    m = (t >= t0) & (t <= t0 + span)
    tt, yy = t[m] - t0, y[m]
    amp = target - pre
    if abs(amp) < 1e-9:
        return None
    frac = lambda f: pre + f * amp
    sgn = np.sign(amp)
    hit = lambda f: np.argmax(np.sign(yy - frac(f)) == sgn) if np.any(
        np.sign(yy - frac(f)) == sgn) else -1
    c10, c90 = hit(0.1), hit(0.9)
    out = {}
    out["delay_ms"] = tt[c10] * 1e3 if c10 >= 0 else float("nan")
    out["rise_ms"] = (tt[c90] - tt[c10]) * 1e3 if c10 >= 0 and c90 >= c10 else float("nan")
    out["overshoot"] = ((yy.max() - target) if amp > 0 else (target - yy.min())) / abs(amp)
    inside = np.abs(yy - target) <= 0.02 * abs(amp)
    settle = float("nan")
    for i in range(len(inside) - 1, -1, -1):
        if not inside[i]:
            settle = tt[i + 1] * 1e3 if i + 1 < len(tt) else float("nan")
            break
    out["settle_ms"] = settle
    out["ss_err"] = float(np.mean(np.abs(yy[-200:] - target)))  # 末 200ms
    return out


def main(csv_path, out_dir):
    d = load(csv_path)
    t = d["t"]

    # ---- 图1:vx 阶跃族 ----
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    m = (t >= 0.4) & (t <= 4.2)
    ax.plot(t[m], d["cmd_vx"][m], "k--", lw=1, alpha=0.6, label="cmd(阶跃,原始)")
    ax.plot(t[m], d["ramp_vx"][m], "C0-", lw=1.5, label="ramp(伺服斜坡,8m/s²)")
    ax.plot(t[m], d["meas_vx"][m], "C3-", lw=1.5, label="meas(实测)")
    for ts in (0.6, 1.6, 2.6, 3.6):
        ax.axvline(ts, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("vx (m/s)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    ax.set_title("底盘阶跃速度响应:vx 阶跃族(+1 / +4 / −2 / 0)")
    ax2.plot(t[m], d["meas_vx"][m] - d["ramp_vx"][m], "C1-", lw=1, label="meas−ramp")
    ax2.plot(t[m], d["meas_vx"][m] - d["cmd_vx"][m], "C2-", lw=1, alpha=0.7, label="meas−cmd")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("误差 (m/s)"); ax2.set_xlabel("t (s)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/step_resp_vx.png", dpi=130)

    # ---- 图2:vy 阶跃(舵翻转)+ wz 阶跃 ----
    fig, (axv, axw) = plt.subplots(2, 1, figsize=(11, 7))
    m = (t >= 4.3) & (t <= 5.8)
    axv.plot(t[m], d["cmd_vy"][m], "k--", lw=1, alpha=0.6, label="cmd vy(阶跃)")
    axv.plot(t[m], d["ramp_vy"][m], "C0-", lw=1.5, label="ramp vy")
    axv.plot(t[m], d["meas_vy"][m], "C3-", lw=1.5, label="meas vy")
    axs = axv.twinx()
    for nm, c in zip(("fl", "fr", "rl", "rr"), ("C1", "C2", "C4", "C5")):
        axs.plot(t[m], np.degrees(d[f"steer_{nm}"][m]), c, lw=1, alpha=0.8, ls="-.")
    axs.set_ylabel("舵角 (°)", color="gray")
    axv.set_ylabel("vy (m/s)"); axv.set_xlabel("t (s)")
    axv.set_title("vy 阶跃 +1.5 m/s:侧向跟踪与四角舵转 90°(点划线,右轴)")
    axv.legend(loc="upper left"); axv.grid(alpha=0.3)
    m = (t >= 6.3) & (t <= 7.8)
    axw.plot(t[m], d["cmd_wz"][m], "k--", lw=1, alpha=0.6, label="cmd wz(阶跃)")
    axw.plot(t[m], d["ramp_wz"][m], "C0-", lw=1.5, label="ramp wz")
    axw.plot(t[m], d["meas_wz"][m], "C3-", lw=1.5, label="meas wz")
    axw.set_ylabel("wz (rad/s)"); axw.set_xlabel("t (s)")
    axw.set_title("wz 阶跃 +1.5 rad/s:角速度跟踪")
    axw.legend(loc="upper left"); axw.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/step_resp_lat.png", dpi=130)

    # ---- metrics(实测 vs 原始阶跃指令;pre=阶跃前 200ms 实测均值)----
    segs = [
        ("vx→+1.0", 0.6, 1.0, d["meas_vx"]),
        ("vx→+4.0", 1.6, 4.0, d["meas_vx"]),
        ("vx→−2.0", 2.6, -2.0, d["meas_vx"]),
        ("vx→0.0", 3.6, 0.0, d["meas_vx"]),
        ("vy→+1.5", 4.6, 1.5, d["meas_vy"]),
        ("wz→+1.5", 6.6, 1.5, d["meas_wz"]),
    ]
    print(f"{'段':10s} {'延迟ms':>7s} {'上升ms':>7s} {'超调%':>6s} "
          f"{'入带ms':>7s} {'稳态误差':>8s}")
    for name, t0, tgt, meas in segs:
        pre = meas[(t >= t0 - 0.25) & (t < t0 - 0.05)].mean()
        mm = metrics(t, meas, t0, tgt, pre)
        if mm:
            print(f"{name:10s} {mm['delay_ms']:7.0f} {mm['rise_ms']:7.0f} "
                  f"{mm['overshoot']*100:6.1f} {mm['settle_ms']:7.0f} "
                  f"{mm['ss_err']:8.3f}")
    print("saved:", f"{out_dir}/step_resp_vx.png", ",", f"{out_dir}/step_resp_lat.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/reports/step_resp.csv",
         sys.argv[2] if len(sys.argv) > 2 else "eval/reports")
