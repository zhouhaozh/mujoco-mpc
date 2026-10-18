#!/usr/bin/env python3
# MIT 单机仿真跟踪曲线(2026-10-18):读 mit_motor_sim 三份 CSV
# (step/chirp/swing),每场景一图三行——①位置跟踪(参考+四变体)
# ②跟踪误差 ③力矩(四变体+理想前馈);打印指标表:
# step=各保持段稳态下垂(重力/kp 刚度),chirp=逐秒窗互相关滞后→相位滞后,
# swing=各快挥段峰值误差,全部变体含 RMS/最大误差。
# 变体:A=纯MIT(真机现状) B=+速度环积分(供应商假设) C=+理想τ前馈
# (T位假设) D=前馈+积分。用法:plot_mit_motor.py [csv_dir] [out_dir]
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

VAR = [("a", "A 纯MIT(现机)", "C0", "-"),
       ("b", "B +速度环积分(固件假设)", "C1", "-"),
       ("c", "C +理想τ前馈(T位假设)", "C2", "-"),
       ("d", "D 前馈+积分(虚线;理想前馈下积分无差可积,与C几乎重合)", "C3", "--")]
SUB = {"step": "软阶跃+保持(重力下垂)", "chirp": "变频正弦 0.5→4Hz(相位滞后)",
       "swing": "三段快挥 Δ1.2rad T=0.30/0.20/0.14s"}


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def rms_max(t, err, skip=0.3):
    m = t >= skip
    return float(np.sqrt(np.mean(err[m] ** 2))), float(np.max(np.abs(err[m])))


def lag_est(ref, y):
    """互相关滞后(s):y(t)≈ref(t−lag),lag>0=y 滞后;1ms 网格,±0.3s 搜索。"""
    r = ref - ref.mean()
    best, bl = -1e18, 0.0
    n = len(r)
    for L in range(-300, 301):
        yy = y - y.mean()
        if L >= 0:
            a, bb = (r[:n - L] if L else r), yy[L:]
        else:
            a, bb = r[-L:], yy[:n + L]
        c = float(np.dot(a, bb))
        if c > best:
            best, bl = c, L * 0.001
    return bl


def fig_scenario(d, scen, out_dir):
    t, qr = d["t"], d["q_ref"]
    fig, (ax1, axv, ax2, ax3) = plt.subplots(
        4, 1, figsize=(11, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    ax1.plot(t, qr, "k--", lw=1.6, label="q_ref")
    axv.plot(t, d["dq_ref"], "k--", lw=1.4, label="dq_ref")
    for tag, lab, c, ls in VAR:
        ax1.plot(t, d[f"q_{tag}"], c, ls=ls, lw=1.3, alpha=0.9, label=lab)
        axv.plot(t, d[f"dq_{tag}"], c, ls=ls, lw=1.0)
        ax2.plot(t, d[f"err_{tag}"], c, ls=ls, lw=1.0, label=lab)
        ax3.plot(t, d[f"tau_{tag}"], c, ls=ls, lw=1.0)
    ax3.plot(t, d["tau_ff_ideal"], "k:", lw=1.2, label="τ_ff 前馈(随帧)")
    ax1.set_ylabel("位置 (rad)"); ax1.legend(loc="best", fontsize=9); ax1.grid(alpha=0.3)
    axv.set_ylabel("速度 (rad/s)"); axv.grid(alpha=0.3)
    ax2.set_ylabel("误差 q−q* (rad)"); ax2.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.5)
    ax3.set_ylabel("力矩 (N·m)"); ax3.set_xlabel("t (s)")
    ax3.grid(alpha=0.3); ax3.legend(loc="best", fontsize=9)
    ax1.set_title(f"MIT 单机仿真:{SUB[scen]}  "
                  "(RS04·J2级负载 kp50/kd30·250Hz帧·9ms延迟)")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/mit_{scen}.png", dpi=130)
    plt.close(fig)


def main(csv_dir, out_dir):
    print(f"{'场景':6s} {'变体':34s} {'RMS rad':>8s} {'max rad':>8s} {'速度RMS':>8s}")
    for scen in ("step", "chirp", "swing"):
        d = load(f"{csv_dir}/mit_{scen}.csv")
        t, qr = d["t"], d["q_ref"]
        fig_scenario(d, scen, out_dir)
        errs = {}
        for tag, lab, _, _ls in VAR:
            err = d[f"err_{tag}"]
            errs[tag] = err
            r, mx = rms_max(t, err)
            vr = float(np.sqrt(np.mean((d[f"dq_{tag}"][t >= 0.3] - d["dq_ref"][t >= 0.3]) ** 2)))
            print(f"{scen:6s} {lab:34s} {r:8.4f} {mx:8.4f} {vr:8.4f}")
        if scen == "step":  # 各保持段末 300ms 稳态下垂(°)
            print("  保持段稳态误差(°, 末300ms均值):")
            for name, t1 in (("hold0.6@0.7-1.0s", 1.0), ("hold−0.4@2.7-3.0s", 3.0),
                             ("hold0.6@4.7-5.0s", 5.0), ("hold0.1@5.7-6.0s", 6.0)):
                row = " ".join(f"{tag}:{np.degrees(np.mean(errs[tag][(t >= t1 - 0.3) & (t <= t1 - 0.01)])):6.2f}"
                               for tag, _, _, _ in VAR)
                print(f"    {name:24s} {row}")
        if scen == "chirp":  # 逐秒窗相位滞后
            print("  相位滞后(°, 逐1s窗互相关):")
            print(f"    {'f(Hz)':>6s} " + " ".join(f"{tag + ':':>9s}" for tag, _, _, _ in VAR))
            for i in range(8):
                m = (t >= i) & (t < i + 1)
                f = 0.5 + 3.5 * (i + 0.5) / 8.0
                row = " ".join(
                    f"{360 * f * lag_est(qr[m], d[f'q_{tag}'][m]):9.1f}"
                    for tag, _, _, _ in VAR)
                print(f"    {f:6.2f} {row}")
        if scen == "swing":  # 各快挥段峰值误差(段+0.2s尾,峰值常在段后)
            print("  快挥段峰值|误差|(°, 段+0.2s尾):")
            for name, t0, t1 in (("T=0.30s", 0.8, 1.3), ("T=0.20s", 2.2, 2.6),
                                 ("T=0.14s", 3.4, 3.74)):
                m = (t >= t0) & (t <= t1)
                row = " ".join(f"{tag}:{np.degrees(np.max(np.abs(errs[tag][m]))):6.2f}"
                               for tag, _, _, _ in VAR)
                print(f"    {name:10s} {row}")
    print("saved:", ", ".join(f"{out_dir}/mit_{s}.png" for s in ("step", "chirp", "swing")))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/reports",
         sys.argv[2] if len(sys.argv) > 2 else "eval/reports")
