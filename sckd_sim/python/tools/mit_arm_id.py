#!/usr/bin/env python3
# 5关节臂 MIT 前馈辨识管线(2026-10-20):与 cpp/apps/mit_arm_id_sim.cpp 配套,
# 也可直接吃真机日志(CSV 列契约相同;_true_ 诊断列缺失时自动跳过验证段)。
# 阶段(--stage,缺省 all):
#   fk     numpy FK(解析 mjcf 链)重力基 vs CSV _true_grav 交叉验证(防符号错)
#   grav   poses 稳态窗 kp·e_p 点测重力 → 双向趋近配对平均(消 ±τc 带)→
#          质量参数最小二乘(重力对连杆质量线性);J1 竖直轴重力≡0=数据金丝雀;
#          附姿位数充分性扫描(--pose-counts)
#   fric   constvel 巡航段 τ̂−ĝ 对 q̇ 回归 → b_j/τc_j
#   inert  chirp:1ms 网格 τ 重构(9ms 帧延迟对齐,250Hz 遥测插值)→ 验证
#          R²/偏差;SG 零相位微分 q̈;去重力摩擦后按惯量线性基回归 Ĵ_i(q)
#          (位形依赖,含 armature 项);--misalign 出错位-参数偏差曲线;
#          饱和样本按 |τ̂|>0.95·T-N 包络剔除
#   ff     swing 参考列 + 辨识模型 → ff_swing.csv(喂回 C++ swing 的 C_id)
#   report swing 三变体闭环验收表
# 用法:mit_arm_id.py [csv_dir] [--out DIR] [--stage all] [--pose-counts 8,16,32]
#        [--misalign] [--model PATH]
# 环境要求:系统 python3(numpy/matplotlib,无 scipy/mujoco)。
import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# ---------- matplotlib(Agg + Noto CJK,同 plot_mit_motor.py 样板) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wenyi/wenyihei.ttf"):
    if os.path.exists(_p):
        font_manager.fontManager.addfont(_p)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 电机常量(与 C++ 侧一致,motors.json) ----------
TN = {
    0: ([0.0, 17.099, 19.8, 23.401, 27.0, 28.8, 30.599, 32.401, 33.661, 34.2, 36.0],
        [120.0, 120.0, 110.0, 95.0, 78.0, 60.0, 52.0, 44.0, 32.0, 10.0, 0.0]),
    1: ([0.0, 43.751, 46.876, 50.0, 54.688, 58.594, 62.501, 65.625, 66.876, 67.187, 75.0],
        [36.0, 36.0, 33.0, 30.0, 26.0, 22.0, 18.0, 12.0, 10.0, 5.0, 0.0]),
}
RS = [0, 0, 1, 1, 1]  # J1/J2=RS04, J3/J4/J5=RS06
VMAX = [36., 36., 75., 75., 75.]
TMAX = [120., 120., 36., 36., 36.]


def tn_avail(j, w):
    xs, ys = TN[RS[j]]
    return float(np.interp(w, xs, ys)) if w <= xs[-1] else 0.0


# ---------- CSV io ----------
def load_csv(path):
    """读 mitarm_*.csv:manifest(# 行,可一行多 k=v)→dict,数据→array。"""
    man, body = {}, []
    with open(path) as f:
        for l in f:
            if l.startswith("#"):
                for tok in l[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        man[k] = v
            else:
                body.append(l)
    d = np.genfromtxt(io.StringIO("".join(body)), delimiter=",", names=True)
    return man, d


def find_model(override=None):
    if override:
        return override
    for base in (os.getcwd(), os.path.join(os.getcwd(), ".."),
                 os.path.dirname(os.path.dirname(os.path.dirname(
                     os.path.abspath(__file__))))):
        p = os.path.join(base, "models/generated/x4_tennis_scene.mjcf")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到 models/generated/x4_tennis_scene.mjcf")


# ---------- FK(解析 mjcf 链:Link1..Link5) ----------
def quat2r(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def trans(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


class Chain:
    def __init__(self, mjcf_path):
        root = ET.parse(mjcf_path).getroot()
        bodies, parent = {}, {}

        def walk(elem, pname):
            for b in elem.findall("body"):
                bodies[b.get("name")] = b
                parent[b.get("name")] = pname
                walk(b, b.get("name"))

        walk(root.find("worldbody"), None)
        chain, cur = [], "Link5"
        while cur is not None and cur != "base_link":
            chain.append(cur)
            cur = parent[cur]
        chain.reverse()  # Link1..Link5(base_link 的 freejoint 不进链)
        self.links = []
        for name in chain:
            b = bodies[name]
            ine = b.find("inertial")
            jnt = b.find("joint")
            q = [float(x) for x in ine.get("quat", "1 0 0 0").split()]
            lk = dict(
                name=name,
                pos=np.array([float(x) for x in b.get("pos").split()]),
                axis=np.array([float(x) for x in jnt.get("axis").split()]),
                com=np.array([float(x) for x in ine.get("pos").split()]),
                mass=float(ine.get("mass")),
                diag=np.array([float(x) for x in ine.get("diaginertia").split()]),
            )
            R = quat2r(q)
            lk["I"] = R @ np.diag(lk["diag"]) @ R.T  # 主轴系→体坐标全张量
            self.links.append(lk)
        self.G = np.array([0.0, 0.0, -9.81])

    def fk(self, q):
        Ts, T = [], np.eye(4)  # base_link 系(高度不影响重力矩)
        for k, lk in enumerate(self.links):
            T = T @ trans(lk["pos"]) @ rot(lk["axis"], q[k])
            Ts.append(T.copy())
        return Ts

    def geom(self, q):
        """锚点 O_i、世界轴 â_i、质心世界坐标。"""
        Ts = self.fk(q)
        aa = [(T[:3, 3], T[:3, :3] @ self.links[k]["axis"])
              for k, T in enumerate(Ts)]
        cs = [Ts[k] @ np.append(lk["com"], 1.0) for k, lk in enumerate(self.links)]
        return aa, cs

    def grav(self, q, masses=None):
        """保持力矩(=−重力对关节轴之矩;与 mj_inverse qfrc_inverse 同号)。"""
        if masses is None:
            masses = [lk["mass"] for lk in self.links]
        aa, cs = self.geom(q)
        g = np.zeros(5)
        for i in range(5):
            oi, ai = aa[i]
            for k in range(i, 5):
                g[i] -= masses[k] * np.dot(ai, np.cross(cs[k][:3] - oi, self.G))
        return g

    def grav_feats(self, q):
        aa, cs = self.geom(q)
        phi = np.zeros((5, 5))
        for i in range(5):
            oi, ai = aa[i]
            for k in range(i, 5):
                phi[i, k] = -np.dot(ai, np.cross(cs[k][:3] - oi, self.G))
        return phi

    def inert_feats(self, q):
        """M_ii 线性基(对角;保留给旧调用)。"""
        d, c = self._pair_feats(q, diag_only=True)
        return d, c

    def mass_matrix_feats(self, q):
        """全矩阵 M_ij 线性基(对称存 i≤j):
        M_ij = Σ_k (m_k·dm[i,j,k] + I6_k·cm[i,j,k]) + δ_ij·arm_i。
        k 遍历两关节共同远端连杆(k≥max(i,j))。"""
        return self._pair_feats(q, diag_only=False)

    def _pair_feats(self, q, diag_only):
        Ts = self.fk(q)
        aa, cs = self.geom(q)
        dm = np.zeros((5, 5, 5))
        cm = np.zeros((5, 5, 5, 6))
        for i in range(5):
            oi, ai = aa[i]
            for j in range(i, 5):
                if diag_only and j != i:
                    continue
                oj, aj = aa[j]
                for k in range(j, 5):
                    rik = cs[k][:3] - oi
                    rjk = cs[k][:3] - oj
                    dm[i, j, k] = np.dot(np.cross(ai, rik), np.cross(aj, rjk))
                    vi = Ts[k][:3, :3].T @ ai
                    vj = Ts[k][:3, :3].T @ aj
                    cm[i, j, k] = [vi[0] * vj[0], vi[1] * vj[1], vi[2] * vj[2],
                                   vi[0] * vj[1] + vi[1] * vj[0],
                                   vi[0] * vj[2] + vi[2] * vj[0],
                                   vi[1] * vj[2] + vi[2] * vj[1]]
        return dm, cm

    def mmat(self, q, th):
        """45 参数(=[m5|I6×5|arm5])→ 5×5 惯量阵。"""
        dm, cm = self.mass_matrix_feats(q)
        M = np.zeros((5, 5))
        for i in range(5):
            for j in range(i, 5):
                v = float(dm[i, j] @ th[:5])
                for k in range(5):
                    v += float(th[5 + 6 * k: 5 + 6 * k + 6] @ cm[i, j, k])
                if i == j:
                    v += th[40 + i]
                M[i, j] = M[j, i] = v
        return M


# ---------- τ 重构(1ms 网格,9ms 帧延迟对齐,遥测线性插值) ----------
def reconstruct(d, delay_ms, frame_ms, shift_ms=0):
    """生效帧=最后发送时刻 ≤ t−(delay−shift) 的帧(ZOH);q/dq 由 250Hz
    遥测样本线性插值。返回 (t, tau[N,5], ok[N,5](非饱和))。"""
    t = d["t"]
    ms = np.round(t * 1000).astype(int)
    send_idx = np.where(ms % frame_ms == 0)[0]
    ts = t[send_idx]
    eff = np.clip(np.searchsorted(ts, t - (delay_ms - shift_ms) * 1e-3,
                                  side="right") - 1, 0, len(send_idx) - 1)
    tau = np.zeros((len(t), 5))
    ok = np.zeros((len(t), 5), bool)
    for j in range(5):
        p = d[f"p_cmd_j{j+1}"][send_idx][eff]
        v = d[f"v_cmd_j{j+1}"][send_idx][eff]
        kp = d[f"kp_cmd_j{j+1}"][send_idx][eff]
        kd = d[f"kd_cmd_j{j+1}"][send_idx][eff]
        tf = d[f"tff_cmd_j{j+1}"][send_idx][eff]
        q = np.interp(t, ts, d[f"q_j{j+1}"][send_idx])
        dq = np.interp(t, ts, d[f"dq_j{j+1}"][send_idx])
        tau[:, j] = kp * (p - q) + kd * (v - dq) + tf
        avail = np.array([tn_avail(j, abs(w)) for w in dq])
        ok[:, j] = np.abs(tau[:, j]) <= 0.95 * avail
    return t, tau, ok


def sg_diff(x, dt, win=21, order=3):
    """Savitzky-Golay 零相位微分(对称窗;无 scipy)。

    窗口拟合一元多项式,导数=一次项系数 → pinv(A) 第 1 行即卷积权重。
    """
    half = win // 2
    idx = np.arange(-half, half + 1) * dt
    A = np.vander(idx, order + 1, increasing=True)  # win×(order+1)
    w = np.linalg.pinv(A)[1]                        # c₁ 行
    return np.convolve(x, w[::-1], mode="same")     # 相关=卷积翻转核


# ---------- 参数持久化(跨进程阶段链) ----------
PARAMS_F = "mitarm_id_params.npz"


def params_path(csv_dir):
    return os.path.join(csv_dir, PARAMS_F)


def save_params(csv_dir, **kw):
    old = {}
    p = params_path(csv_dir)
    if os.path.exists(p):
        old = dict(np.load(p))
    old.update({k: v for k, v in kw.items() if v is not None})
    np.savez(p, **old)


def load_params(csv_dir):
    p = params_path(csv_dir)
    if not os.path.exists(p):
        return {}
    return {k: np.load(p)[k] for k in np.load(p).files}


# ---------- 各阶段 ----------
def stage_fk(chain, d_poses):
    """在 poses 稳态窗验证(q 静止,遥测 14bit 量化 → 阈 0.1 N·m;
    符号/几何错会给出 ~2|g| 量级偏差,阈值的选取只受量化限制)。"""
    sel = np.where(d_poses["settled"] == 1)[0][::7]
    err = 0.0
    for i in sel:
        q = [d_poses[f"q_j{j+1}"][i] for j in range(5)]
        g = chain.grav(q)
        gt = np.array([d_poses[f"grav_j{j+1}"][i] for j in range(5)])
        err = max(err, np.abs(g - gt).max())
    ok = err < 0.1
    print(f"[fk] numpy 重力基 vs mj_inverse 真值列({len(sel)} 稳态姿位):"
          f"max|Δg|={err:.3f} N·m({'通过(<0.1)' if ok else ' FAIL(符号/几何约定?)'})")
    return ok


def autodetect_settled(d):
    """settled 列缺失时从遥测自动判稳(判据与 C++ 侧一致):
    过去 300ms 内 max|dq|<2e-2 且逐关节 std(kp·e_p)<0.3 → 之后 0.5s 为
    记录窗;跳过记录窗后重新等稳态。"""
    n = len(d["t"])
    kpe = np.stack([d[f"kp_cmd_j{j+1}"] * (d[f"p_cmd_j{j+1}"] - d[f"q_j{j+1}"])
                    for j in range(5)], 1)
    vmax = np.stack([np.abs(d[f"dq_j{j+1}"]) for j in range(5)], 1).max(1)
    dt = float(d["t"][1] - d["t"][0])
    W, R = max(2, int(round(0.3 / dt))), max(2, int(round(0.5 / dt)))
    settled = np.zeros(n, bool)
    i = W
    while i < n:
        w = slice(i - W, i)
        if vmax[w].max() < 2e-2 and (kpe[w].std(0) < 0.3).all():
            e = min(n, i + R)
            settled[i:e] = True
            i = e + W
            continue
        i += 10
    return settled


def stage_grav(chain, d_poses, pose_counts):
    # 连续 settled==1 段=稳态记录窗(每姿位×每方向一段)
    if "settled" in d_poses.dtype.names:
        s = d_poses["settled"] == 1
    else:  # 真机日志无 settled 列 → 自动判稳
        s = autodetect_settled(d_poses)
        print(f"[grav] 日志无 settled 列,自动判稳命中 {int(s.sum())} 行")
    edges = np.diff(np.concatenate([[0], s.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    qall = np.stack([d_poses[f"q_j{j+1}"] for j in range(5)], 1)
    tau = np.stack([d_poses[f"kp_cmd_j{j+1}"] * (d_poses[f"p_cmd_j{j+1}"] - d_poses[f"q_j{j+1}"]) +
                    d_poses[f"kd_cmd_j{j+1}"] * (d_poses[f"v_cmd_j{j+1}"] - d_poses[f"dq_j{j+1}"]) +
                    d_poses[f"tff_cmd_j{j+1}"] for j in range(5)], 1)
    runs = []
    for a, b in zip(starts, ends):
        if b - a < 50:
            continue
        runs.append((qall[a:b].mean(0), tau[a:b].mean(0),
                     int(np.sign(d_poses["pass_dir"][a:b].mean()))))
    print(f"[grav] 稳态窗 {len(runs)}"
          f"(+{sum(1 for r in runs if r[2] > 0)}/−{sum(1 for r in runs if r[2] < 0)})")
    canary = np.abs(np.mean([r[1][0] for r in runs]))
    print(f"[grav] J1 竖直轴重力金丝雀 |mean τ̂₁|={canary:.3f} N·m"
          f"({'通过(<0.1)' if canary < 0.1 else ' FAIL'})")
    # 双向配对(正/倒序两遍各趋近一次;平均消 ±τc 静摩擦带)
    pos_r = [r for r in runs if r[2] > 0]
    neg_r = [r for r in runs if r[2] < 0]
    used, pairs = set(), []
    for r in pos_r:
        if not neg_r:
            break
        dist = [np.linalg.norm(n[0] - r[0]) for n in neg_r]
        jj = int(np.argmin(dist))
        if jj in used or min(dist) >= 0.05:
            continue
        used.add(jj)
        pairs.append(((r[0] + neg_r[jj][0]) / 2, (r[1] + neg_r[jj][1]) / 2))
    qp = np.array([p[0] for p in pairs])
    tp = np.array([p[1] for p in pairs])
    print(f"[grav] 双向配对姿位 {len(qp)}(未配对窗 {len(runs) - 2 * len(qp)})")

    def gX(qs):
        X = np.zeros((5 * len(qs), 5))
        for n, q in enumerate(qs):
            X[5 * n:5 * n + 5] = chain.grav_feats(q)
        return X

    X = gX(qp)
    Y = tp.reshape(-1)
    # 串联链质量单独不可辨识(经典结论:只有集中组合可辨识)→ SVD 秩分析+伪逆
    U, sv, Vt = np.linalg.svd(X, full_matrices=False)
    rank = int((sv > sv[0] * 1e-6).sum())
    theta = Vt.T @ (U.T @ Y / np.where(sv > sv[0] * 1e-6, sv, np.inf))
    resid = Y - X @ theta
    mtruth = [lk["mass"] for lk in chain.links]
    print(f"[grav] 回归:秩 {rank}/5,条件数 {sv[0] / sv[rank - 1]:.1f}"
          f"(质量参数仅集中组合可辨识,看 ĝ 精度而非质量值)")
    print("[grav] 质量参数(量级参考,非唯一):" +
          " ".join(f"m{k+1}={theta[k]:+.2f}" for k in range(5)) +
          " | mjcf 真值 " + " ".join(f"{m:.2f}" for m in mtruth))
    print(f"[grav] 拟合残差 RMS={np.sqrt((resid ** 2).mean()):.3f} N·m"
          f" max={np.abs(resid).max():.3f}(点测峰值 {np.abs(tp).max():.1f} N·m)")
    # 留出验证:偶数位姿拟合→奇数位姿评估(deployable 精度)
    if len(qp) >= 6:
        ev = np.arange(1, len(qp), 2)
        tr = np.arange(0, len(qp), 2)
        Xt = gX(qp[tr])
        Yt = np.concatenate([tp[i] for i in tr])  # 姿位主序:每姿位 5 关节方程
        th2 = np.linalg.pinv(Xt, rcond=1e-6) @ Yt
        err = np.array([chain.grav(q, th2) for q in qp[ev]]) - tp[ev]
        print(f"[grav] 留出验证(半分):ĝ RMSE={np.sqrt((err ** 2).mean()):.3f} N·m"
              f" max={np.abs(err).max():.3f}")
    if pose_counts:
        print("[grav] 姿位数充分性(前 K 配对拟合,全体配对评估):")
        for K in pose_counts:
            K = min(K, len(qp))
            if K < 2:
                continue
            thk = np.linalg.pinv(X[:5 * K], rcond=1e-6) @ Y[:5 * K]
            err = np.array([chain.grav(q, thk) for q in qp]) - tp
            print(f"   K={K:3d}: 全姿位重力 RMSE={np.sqrt((err ** 2).mean()):.3f} N·m")
    return theta


def stage_fric(chain, d_cv, gtheta, man):
    # τ 必须用延迟对齐重构(运动中行内回显比驱动器实际生效帧新 9ms,
    # 未对齐会多出 kp·v·delay 的系统斜率→b 虚高)。
    # 重力斜坡下 droop 随 g(q) 变化:q̇=v/(1+g′/kp),q̈≈v²·g″/kp≠0,
    # M·q̈ 污染摩擦拟合且 ∝v²——回归加 q̈ 列(J_eff 吸收,兼作惯量互验)。
    delay = int(man.get("delay_ms", 9))
    frame = int(man.get("frame_ms", 4))
    t, tau_rec, _ = reconstruct(d_cv, delay, frame)
    dt = float(t[1] - t[0])
    qs = np.stack([d_cv[f"q_j{k+1}"] for k in range(5)], 1)
    b_hat, tc_hat = np.zeros(5), np.zeros(5)
    for j in range(5):
        vcmd = d_cv[f"v_cmd_j{j+1}"]
        aref = d_cv[f"aref_j{j+1}"]
        dq = np.interp(t, t, d_cv[f"dq_j{j+1}"])
        ddq = sg_diff(dq, dt, win=11)
        mask = (np.abs(aref) < 1e-9) & (np.abs(dq) > 0.02) & (np.abs(vcmd) > 0.05)
        if mask.sum() < 100:
            print(f"[fric] j{j+1}: 巡航样本不足({int(mask.sum())})")
            continue
        ghat = np.array([chain.grav(q, gtheta)[j] for q in qs])
        y = tau_rec[:, j] - ghat
        x = dq
        A = np.stack([x, np.sign(x), ddq], 1)
        sol, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        b_hat[j], tc_hat[j] = sol[0], sol[1]
        r = y[mask] - A[mask] @ sol
        print(f"[fric] j{j+1}: b={sol[0]:+.4f} τc={sol[1]:+.3f} J_eff={sol[2]:+.3f}"
              f" 残差RMS={np.sqrt((r ** 2).mean()):.3f} N·m 样本={int(mask.sum())}")
    return b_hat, tc_hat


def stage_inert(chain, d_chirp, gtheta, b_hat, tc_hat, man, misalign=False):
    delay = int(man.get("delay_ms", 9))
    frame = int(man.get("frame_ms", 4))
    t, tau, ok = reconstruct(d_chirp, delay, frame)
    if "tau_j2" in d_chirp.dtype.names:
        taut = np.stack([d_chirp[f"tau_j{j+1}"] for j in range(5)], 1)
        r2s, means = [], []
        for j in range(5):
            m = ok[:, j]
            e = tau[m, j] - taut[m, j]
            r2 = 1 - np.sum(e ** 2) / np.sum((taut[m, j] - taut[m, j].mean()) ** 2)
            r2s.append(r2)
            means.append(np.abs(e.mean()))
        print("[recon] 对齐重构 vs 真实施加力矩:R²=" +
              " ".join(f"j{j+1}={r2s[j]:.5f}" for j in range(5)))
        print(f"[recon] max|均值偏差|={max(means):.4f} N·m"
              f"({'通过(<0.5%量程=0.6)' if max(means) < 0.6 else ' 偏大'})")
    dt = float(t[1] - t[0])
    qs = np.stack([d_chirp[f"q_j{j+1}"] for j in range(5)], 1)
    dqs = np.stack([d_chirp[f"dq_j{j+1}"] for j in range(5)], 1)
    ddqs = np.stack([sg_diff(dqs[:, j], dt) for j in range(5)], 1)
    ghat = np.array([chain.grav(q, gtheta) for q in qs])
    fhat = b_hat * dqs + tc_hat * np.tanh(dqs / 0.05)
    stride = 4  # 全矩阵特征较重,抽样 ~10k
    smp = np.arange(0, len(qs), stride)
    # 特征只依赖 q:预计算后错位扫描只换重构/残差(τ、mask)
    qm = qs[smp]
    FM = np.zeros((len(qm), 5, 5, 5))      # [s, i, j, k] 质量特征(对称展开)
    FI = np.zeros((len(qm), 5, 5, 5, 6))   # [s, i, j, k, 6] 惯量特征
    for s, q in enumerate(qm):
        dm, cm = chain.mass_matrix_feats(q)
        for i in range(5):
            for j in range(5):
                a2, b2 = (i, j) if i <= j else (j, i)
                FM[s, i, j] = dm[a2, b2]
                FI[s, i, j] = cm[a2, b2]
    A = ddqs[smp]

    def fit(shift=0):
        _, tau_s, ok_s = reconstruct(d_chirp, delay, frame, shift)
        res = tau_s - ghat - fhat
        Xs, ys = [], []
        for i in range(5):
            m = ok_s[smp, i] & (np.abs(A[:, i]) > 0.5)
            if m.sum() < 100:
                continue
            X = np.zeros((int(m.sum()), 45))
            # τ_i = Σ_j M_ij·q̈_j:质量/惯量特征按 q̈_j 加权求和(全耦合)
            X[:, 0:5] = (A[m][:, :, None] * FM[m, i]).sum(1)
            X[:, 5:35] = (A[m][:, :, None, None] * FI[m, i]).sum(1).reshape(-1, 30)
            X[:, 40 + i] = A[m, i]
            Xs.append(X)
            ys.append(res[smp, i][m])
        XA, ya = np.vstack(Xs), np.concatenate(ys)
        # 截断伪逆:45 参数含大量不可辨识组合(同重力阶),全秩解会 1e6 级发散
        th = np.linalg.pinv(XA, rcond=1e-4) @ ya
        rr = ya - XA @ th
        return th, np.sqrt((rr ** 2).mean()), len(ya)

    th, rms, n = fit(0)
    print(f"[inert] 全矩阵耦合回归残差 RMS={rms:.3f} N·m(样本 {n})")
    idx = np.arange(0, len(qs), 20)
    Jh = np.array([chain.mmat(q, th) for q in qs[idx]])
    if "Mii_j2" in d_chirp.dtype.names:
        Mt = np.stack([d_chirp[f"Mii_j{j+1}"][idx] for j in range(5)], 1)
        for j in range(5):
            e = (Jh[:, j, j] - Mt[:, j]) / max(np.abs(Mt[:, j]).mean(), 1e-9)
            print(f"[inert] j{j+1}: Ĵ_ii 轨迹RMS相对误差={np.sqrt((e ** 2).mean()) * 100:.1f}%"
                  f"(|Mii|均值 {np.abs(Mt[:, j]).mean():.4f} kg·m²)")
    print("[inert] armature(截断伪逆,量级参考):" +
          " ".join(f"arm{j+1}={th[40 + j]:+.4f}" for j in range(5)))
    if misalign:
        print("[misalign] 延迟错位-Ĵ 预测偏差(0 为正确对齐;比参数差有意义):")
        for shift in range(-12, 13, 3):
            ths, _, _ = fit(shift)
            Js = np.array([chain.mmat(q, ths) for q in qs[idx]])
            rel = np.abs(Js - Jh).max((0, 2)) / np.abs(Jh).max((0, 2))
            print(f"   shift={shift:+3d}ms: max|ΔM_ij|/|M|max = " +
                  " ".join(f"j{j+1}={float(rel[j]) * 100:5.1f}%" for j in range(5)))
    return th


def coriolis_ff(chain, q, dq, th):
    """Ĉ(q,q̇)·q̇,Christoffel 符号:∂M/∂q_k 由 mmat 中心差分(±1e-4)。"""
    G = np.zeros((5, 5, 5))  # G[i,j,k]=∂M_ij/∂q_k
    for k in range(5):
        qp, qm = q.copy(), q.copy()
        qp[k] += 1e-4
        qm[k] -= 1e-4
        G[:, :, k] = (chain.mmat(qp, th) - chain.mmat(qm, th)) / 2e-4
    C = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            C[i, j] = 0.5 * (G[i, j, :] @ dq + G[i, :, j] @ dq - G[j, :, i] @ dq)
    return C @ dq


def stage_ff(chain, d_swing, gtheta, b_hat, tc_hat, itheta, csv_dir):
    t = d_swing["t"]
    qs = np.stack([d_swing[f"qref_j{j+1}"] for j in range(5)], 1)
    vs = np.stack([d_swing[f"vref_j{j+1}"] for j in range(5)], 1)
    as_ = np.stack([d_swing[f"aref_j{j+1}"] for j in range(5)], 1)
    ghat = np.array([chain.grav(q, gtheta) for q in qs])
    fhat = b_hat * vs + tc_hat * np.tanh(vs / 0.05)
    ff = np.zeros_like(ghat)
    # 科氏按关节选择:截断参数化的不可辨识方向在 ∂M/∂q 组合里被放大,
    # 只对科氏效应主导且估得准的关节施加(默认 J1;真机标定后按残差复查)
    csel = np.zeros(5, bool)
    for tok in os.environ.get("MITARM_CORIOLIS", "1").split(","):
        tk = tok.strip().lower().lstrip("j")
        if tk in ("1", "2", "3", "4", "5"):
            csel[int(tk) - 1] = True
    print(f"[ff] 科氏项施加关节:" + ",".join(f"J{k+1}" for k in range(5) if csel[k]))
    for r in range(len(qs)):  # 重力+摩擦+M̂q̈(耦合)+选择性 Christoffel 科氏
        cor = coriolis_ff(chain, qs[r], vs[r], itheta) * csel
        ff[r] = ghat[r] + fhat[r] + chain.mmat(qs[r], itheta) @ as_[r] + cor
    out = os.path.join(csv_dir, "ff_swing.csv")
    with open(out, "w") as f:
        f.write("# mit_arm_id.py --stage ff 生成(辨识模型前馈,1ms 网格,"
                "与 mitarm_swing.csv 同参考)\n")
        f.write("t,ff_j1,ff_j2,ff_j3,ff_j4,ff_j5\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.4f}," + ",".join(f"{x:.3f}" for x in ff[i]) + "\n")
    print(f"[ff] {len(t)} 行 -> {out}")
    return out


def stage_report(d_swing, extras=None):
    """extras:真机双遍挥拍 {tag: csv路径}——其 q_j* 列拼成 q_{tag}_j*。"""
    data = {n: d_swing[n] for n in d_swing.dtype.names}
    for tag, dx in (extras or {}).items():
        for j in range(5):
            data[f"q_{tag}_j{j+1}"] = dx[f"q_j{j+1}"]
    tags = [("a", "A纯MIT"), ("ci", "C_ideal"), ("cid", "C_id辨识")]
    have = [t for t, _ in tags if f"q_{t}_j1" in data]
    dt = d_swing["t"][1] - d_swing["t"][0]
    i0 = int(0.3 / dt)
    sw0 = int(1.0 / dt)
    print("\n===== swing 三变体闭环验收(RMS 跳过前 0.3s;峰差=快挥段 1.0s 后)=====")
    met = {}
    for tag, _ in tags:
        if tag not in have:
            continue
        met[tag] = {}
        for j in range(5):
            e = data[f"q_{tag}_j{j+1}"] - data[f"qref_j{j+1}"]
            met[tag][j] = (np.sqrt(np.mean(e[i0:] ** 2)), np.abs(e[sw0:]).max())
    print("      " + "".join(f"{dict(tags)[t]:>20}" for t in have))
    for j in range(5):
        row = f"J{j+1}   "
        for t in have:
            rms, pk = met[t][j]
            row += f" {rms*1e3:7.1f}mrad/{pk*57.3:5.1f}°"
        print(row)
    if all(t in met for t, _ in tags):
        print("比值:")
        for j in range(5):
            ra, ri, rc = (met["a"][j][0], met["ci"][j][0], met["cid"][j][0])
            print(f"J{j+1}: A/C_id={ra / max(rc, 1e-9):5.0f}×   "
                  f"C_id/C_ideal={rc / max(ri, 1e-9):4.1f}×")


def stage_plots(chain, csv_dir, out, pose_counts):
    """三张图:重力点测/姿位数曲线、摩擦回线、swing 三变体误差。"""
    _, dp = load_csv(os.path.join(csv_dir, "mitarm_poses.csv"))
    _, dc = load_csv(os.path.join(csv_dir, "mitarm_constvel.csv"))
    _, ds = load_csv(os.path.join(csv_dir, "mitarm_swing.csv"))
    prev = load_params(csv_dir)
    # 图1:重力
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    s = dp["settled"] == 1
    gt = np.stack([dp[f"grav_j{j+1}"][s] for j in range(5)], 1)
    est = np.stack([dp[f"kp_cmd_j{j+1}"][s] * (dp[f"p_cmd_j{j+1}"][s] - dp[f"q_j{j+1}"][s]) +
                    dp[f"kd_cmd_j{j+1}"][s] * (dp[f"v_cmd_j{j+1}"][s] - dp[f"dq_j{j+1}"][s])
                    for j in range(5)], 1)
    for j in range(5):
        ax[0].scatter(gt[:, j], est[:, j], s=4, label=f"J{j+1}")
    lim = np.abs(gt).max() * 1.1
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax[0].set_xlabel("mj_inverse 重力真值 (N·m)")
    ax[0].set_ylabel("稳态 kp·e_p 点测 (N·m)")
    ax[0].set_title("重力点测 vs 真值(settled 窗)")
    ax[0].legend(fontsize=8, markerscale=2)
    gtheta = prev.get("gtheta")
    ks, rms = [], []
    if gtheta is not None:
        # 复用 stage_grav 的配对逻辑做 K 曲线
        qall = np.stack([dp[f"q_j{j+1}"][s] for j in range(5)], 1)
        e = np.diff(np.concatenate([[0], s.astype(int), [0]]))
        st, en = np.where(e == 1)[0], np.where(e == -1)[0]
        runs = [(qall[a:b].mean(0), est[a:b].mean(0),
                 int(np.sign(dp["pass_dir"][a:b].mean()))) for a, b in zip(st, en)
                if b - a >= 50]
        pos_r = [r for r in runs if r[2] > 0]
        neg_r = [r for r in runs if r[2] < 0]
        pairs, used = [], set()
        for r in pos_r:
            dist = [np.linalg.norm(n[0] - r[0]) for n in neg_r]
            jj = int(np.argmin(dist))
            if neg_r and jj not in used and min(dist) < 0.05:
                used.add(jj)
                pairs.append(((r[0] + neg_r[jj][0]) / 2, (r[1] + neg_r[jj][1]) / 2))
        qp = np.array([p[0] for p in pairs])
        tp = np.array([p[1] for p in pairs])
        X = np.zeros((5 * len(qp), 5))
        for n, q in enumerate(qp):
            X[5 * n:5 * n + 5] = chain.grav_feats(q)
        Y = tp.reshape(-1)
        for K in range(2, len(qp) + 1):
            thk = np.linalg.pinv(X[:5 * K], rcond=1e-6) @ Y[:5 * K]
            err = np.array([chain.grav(q, thk) for q in qp]) - tp
            ks.append(K)
            rms.append(np.sqrt((err ** 2).mean()))
    ax[1].plot(ks, rms, "o-")
    ax[1].set_xlabel("姿位数 K(双向配对)")
    ax[1].set_ylabel("全姿位重力 RMSE (N·m)")
    ax[1].set_title("激励充分性:姿位数扫描")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mitarm_id_grav.png"), dpi=130)
    plt.close(fig)
    # 图2:摩擦回线
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.2), sharey=False)
    man, _ = load_csv(os.path.join(csv_dir, "mitarm_constvel.csv"))
    t, tau_rec, _ = reconstruct(dc, int(man.get("delay_ms", 9)), int(man.get("frame_ms", 4)))
    for j in range(5):
        dq = dc[f"dq_j{j+1}"]
        m = (np.abs(dc[f"aref_j{j+1}"]) < 1e-9) & (np.abs(dq) > 0.02) & \
            (np.abs(dc[f"v_cmd_j{j+1}"]) > 0.05)
        qs_ = np.stack([dc[f"q_j{k+1}"][m] for k in range(5)], 1)
        y = tau_rec[m, j] - np.array([chain.grav(q, gtheta)[j] for q in qs_]) \
            if gtheta is not None else tau_rec[m, j]
        axs[j].scatter(dq[m], y, s=2, alpha=0.3)
        b_, tc_ = prev.get("b_hat", np.zeros(5))[j], prev.get("tc_hat", np.zeros(5))[j]
        xs = np.linspace(-1.1, 1.1, 50)
        axs[j].plot(xs, b_ * xs + tc_ * np.sign(xs), "r-", lw=1.5)
        axs[j].set_title(f"J{j+1}: b={b_:+.3f} τc={tc_:+.2f}", fontsize=9)
        axs[j].set_xlabel("q̇ (rad/s)")
    axs[0].set_ylabel("τ̂−ĝ (N·m)")
    fig.suptitle("摩擦辨识(对齐重构,红线=拟合)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mitarm_id_fric.png"), dpi=130)
    plt.close(fig)
    # 图3:swing 三变体
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    dt = ds["t"][1] - ds["t"][0]
    i0 = int(0.3 / dt)
    colors = {"a": "tab:red", "ci": "tab:green", "cid": "tab:blue"}
    names = {"a": "A 纯MIT", "ci": "C_ideal", "cid": "C_id辨识"}
    for r, j in enumerate((0, 1, 2)):
        for tag in ("a", "ci", "cid"):
            if f"q_{tag}_j{j+1}" not in ds.dtype.names:
                continue
            e = (ds[f"q_{tag}_j{j+1}"] - ds[f"qref_j{j+1}"]) * 57.3
            axs[r].plot(ds["t"][i0:], e[i0:], color=colors[tag], lw=1.0,
                        label=names[tag])
        axs[r].set_ylabel(f"J{j+1} 误差 (°)")
        axs[r].grid(alpha=0.3)
        axs[r].legend(fontsize=8)
    axs[2].set_xlabel("t (s)")
    fig.suptitle("swing 三变体闭环(快挥段 1.0-1.64s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mitarm_id_swing.png"), dpi=130)
    plt.close(fig)
    print(f"[plots] 3 图 -> {out}/mitarm_id_{{grav,fric,swing}}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_dir", nargs="?", default="eval/reports")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["all", "fk", "grav", "fric", "inert", "ff", "report",
                             "plots"])
    ap.add_argument("--pose-counts", default="8,16,32")
    ap.add_argument("--misalign", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--swing-a", default=None,
                    help="真机 FF-off 挥拍日志(列 q_j*,拼为 q_a_j*)")
    ap.add_argument("--swing-cid", default=None,
                    help="真机 FF-on 挥拍日志(列 q_j*,拼为 q_cid_j*)")
    a = ap.parse_args()
    out = a.out or a.csv_dir
    os.makedirs(out, exist_ok=True)
    chain = Chain(find_model(a.model))
    print("[io] 链 " + "→".join(lk["name"] for lk in chain.links) +
          "; 质量 " + ",".join(f"{lk['mass']:.3f}" for lk in chain.links))
    stages = ([a.stage] if a.stage != "all"
              else ["fk", "grav", "fric", "inert", "ff", "report", "plots"])
    prev = load_params(a.csv_dir)
    gtheta = prev.get("gtheta")
    b_hat, tc_hat = prev.get("b_hat"), prev.get("tc_hat")
    itheta = prev.get("itheta")
    for st in stages:
        if st == "fk":
            _, d = load_csv(os.path.join(a.csv_dir, "mitarm_poses.csv"))
            stage_fk(chain, d)
        elif st == "grav":
            _, d = load_csv(os.path.join(a.csv_dir, "mitarm_poses.csv"))
            gtheta = stage_grav(chain, d, [int(x) for x in a.pose_counts.split(",")])
            save_params(a.csv_dir, gtheta=gtheta)
        elif st == "fric":
            if gtheta is None:
                sys.exit("[fric] 需先跑 grav")
            man, d = load_csv(os.path.join(a.csv_dir, "mitarm_constvel.csv"))
            b_hat, tc_hat = stage_fric(chain, d, gtheta, man)
            save_params(a.csv_dir, b_hat=b_hat, tc_hat=tc_hat)
        elif st == "inert":
            if gtheta is None or b_hat is None:
                sys.exit("[inert] 需先跑 grav+fric")
            man, d = load_csv(os.path.join(a.csv_dir, "mitarm_chirp.csv"))
            itheta = stage_inert(chain, d, gtheta, b_hat, tc_hat, man, a.misalign)
            save_params(a.csv_dir, itheta=itheta)
        elif st == "ff":
            if itheta is None:
                sys.exit("[ff] 需先跑 inert")
            _, d = load_csv(os.path.join(a.csv_dir, "mitarm_swing.csv"))
            stage_ff(chain, d, gtheta, b_hat, tc_hat, itheta, a.csv_dir)
        elif st == "report":
            _, d = load_csv(os.path.join(a.csv_dir, "mitarm_swing.csv"))
            extras = {}
            for tag, path in (("a", a.swing_a), ("cid", a.swing_cid)):
                if path:
                    _, dx = load_csv(path)
                    extras[tag] = dx
            stage_report(d, extras)
        elif st == "plots":
            stage_plots(chain, a.csv_dir, out,
                        [int(x) for x in a.pose_counts.split(",")])


if __name__ == "__main__":
    main()
