#!/usr/bin/env python3
# 前馈辨识误差×积分兜底对比(2026-10-18):读多组 mit_motor_sim 输出目录
# (标称=理想前馈;fferr_*=ff_J/ff_G 注入辨识误差),三行图——
# ①swing 位置误差 ②swing 速度误差 ③step 全程误差(重力误差=保持段偏移)
# 叠加各配置 C/D(基线只画 C,D≈C 重合;误差组 C 实线+D 虚线同色;
# B 无前馈做橙色参考),打印指标表:step 保持段下垂/swing 峰差/
# swing 速度峰差/chirp 位置RMS(变体×配置)。
# 用法:mit_ff_err.py out.png 基线目录 标签=目录 [标签=目录 ...]
# 例:mit_ff_err.py eval/reports/mit_ff_err.png eval/reports \
#       "J−20%=eval/reports/fferr_J16" "G+20%=eval/reports/fferr_G96" \
#       "组合=eval/reports/fferr_both"
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

ERRC = ["C4", "C5", "C6"]  # 误差组配色(基线 C=绿 C2,B 参考=C1 橙)
VN = [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]


def load(d, scen):
    return np.genfromtxt(f"{d}/mit_{scen}.csv", delimiter=",", names=True)


def metrics(dirs):
    """指标:step 4 保持段均值最大|下垂|°;swing[0.8,3.74] 峰差°/速度峰差;
    chirp RMS mrad。返回 {标签: {变体: (sag, pk, vpk, rms)}}"""
    out = {}
    for lab, dd in dirs:
        st, sw, ch = load(dd, "step"), load(dd, "swing"), load(dd, "chirp")
        t = st["t"]
        m = {}
        for tag, _ in VN:
            sag = max(abs(np.degrees(np.mean(st[f"err_{tag}"][(t >= t1 - 0.3) & (t <= t1 - 0.01)])))
                      for t1 in (1.0, 3.0, 5.0, 6.0))
            msw = (sw["t"] >= 0.8) & (sw["t"] <= 3.74)
            pk = np.degrees(np.max(np.abs(sw[f"err_{tag}"][msw])))
            vpk = np.max(np.abs(sw[f"dq_{tag}"][msw] - sw["dq_ref"][msw]))
            rms = float(np.sqrt(np.mean(ch[f"err_{tag}"][ch["t"] >= 0.3] ** 2))) * 1e3
            m[tag] = (sag, pk, vpk, rms)
        out[lab] = m
    return out


def main(out_png, base_dir, pairs):
    dirs = [("理想", base_dir)] + pairs
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 11),
                                     gridspec_kw={"height_ratios": [2, 2, 2]})
    sw_b, st_b = load(base_dir, "swing"), load(base_dir, "step")

    # ①swing 位置误差 ②swing 速度误差
    for ax, is_vel, ylab in ((a1, False, "swing 位置误差 q−q* (rad)"),
                             (a2, True, "swing 速度误差 dq−dq* (rad/s)")):
        ax.axhline(0, color="k", lw=0.5)
        eb = sw_b["err_b"] if not is_vel else (sw_b["dq_b"] - sw_b["dq_ref"])
        ax.plot(sw_b["t"], eb, color="C1", lw=0.9, alpha=0.55,
                label="B 无前馈(参考)")
        cb = sw_b["err_c"] if not is_vel else (sw_b["dq_c"] - sw_b["dq_ref"])
        ax.plot(sw_b["t"], cb, color="C2", lw=1.5, label="C 理想前馈")
        for i, (lab, dd) in enumerate(pairs):
            sw = load(dd, "swing")
            c = ERRC[i % len(ERRC)]
            cc = sw["err_c"] if not is_vel else (sw["dq_c"] - sw["dq_ref"])
            dd_ = sw["err_d"] if not is_vel else (sw["dq_d"] - sw["dq_ref"])
            ax.plot(sw["t"], cc, color=c, lw=1.2, label=f"C {lab}")
            ax.plot(sw["t"], dd_, color=c, ls="--", lw=1.2, alpha=0.75,
                    label=f"D {lab}")
        ax.set_ylabel(ylab); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
    # ③step 全程误差(重力误差→保持段偏移)
    a3.axhline(0, color="k", lw=0.5)
    a3.plot(st_b["t"], st_b["err_b"], color="C1", lw=0.9, alpha=0.55,
            label="B 无前馈(参考)")
    a3.plot(st_b["t"], st_b["err_c"], color="C2", lw=1.5, label="C 理想前馈")
    for i, (lab, dd) in enumerate(pairs):
        st = load(dd, "step")
        c = ERRC[i % len(ERRC)]
        a3.plot(st["t"], st["err_c"], color=c, lw=1.2, label=f"C {lab}")
        a3.plot(st["t"], st["err_d"], color=c, ls="--", lw=1.2, alpha=0.75,
                label=f"D {lab}")
    a3.set_ylabel("step 位置误差 (rad)"); a3.set_xlabel("t (s)")
    a3.grid(alpha=0.3); a3.legend(fontsize=8, ncol=3)
    a1.set_title("前馈辨识误差下的积分兜底量化(mit_motor_sim,J=0.2 G=8 kp50/kd30)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    m = metrics(dirs)
    print(f"{'配置':8s} {'变体':4s} {'下垂°':>7s} {'swing峰差°':>9s} "
          f"{'速度峰差':>8s} {'chirp RMS mrad':>13s}")
    for lab, _ in dirs:
        for tag, vn in VN:
            sag, pk, vpk, rms = m[lab][tag]
            print(f"{lab:8s} {vn:4s} {sag:7.2f} {pk:9.2f} {vpk:8.3f} {rms:13.1f}")
    # 一致性自检:A/B 不依赖前馈,各配置应完全一致
    ref = m["理想"]["a"][0]
    dif = max(abs(m[l]["a"][0] - ref) for l, _ in dirs)
    print(f"[check] A 下垂跨配置最大差 {dif:.3f}°(应=0)")
    print("saved:", out_png)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    pairs = []
    for a in sys.argv[3:]:
        eq = a.find("=")
        pairs.append((a[:eq], a[eq + 1:]))
    main(sys.argv[1], sys.argv[2], pairs)
