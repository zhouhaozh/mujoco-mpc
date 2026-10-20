"""Formula-rich method-6 diagrams; no inference, fitting or result mutation.

Render in the Ubuntu project .venv-tennis.  The report supplies an explicit
protocol description when the causal stream experiment is ready.  This module
does not evaluate trajectories or read scoring reference events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .gp_analysis_plots import _style, BLUE, ORANGE, PURPLE, GREEN, GRAY


def _canvas(title, subtitle, rows=4, height=18):
    fig = plt.figure(figsize=(20, height))
    ax = fig.add_axes([.025, .075, .95, .83])
    ax.set(xlim=(0, 20), ylim=(0, rows * 3.6))
    ax.axis("off")
    fig.text(.5, .967, title, ha="center", va="top", fontsize=23, weight="bold")
    fig.text(.5, .929, subtitle, ha="center", va="top", fontsize=12, color=GRAY)
    return fig, ax


def _box(ax, x, y, w, h, title, lines, color=BLUE, fontsize=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=.11", linewidth=1.65,
        edgecolor=color, facecolor=matplotlib.colors.to_rgba(color, .045)))
    ax.text(x+.28, y+h-.28, title, ha="left", va="top", color=color,
            fontsize=17, weight="bold")
    start = y+h-.80
    spacing = (h-1.05)/max(1, len(lines)-1)
    for j, line in enumerate(lines):
        ax.text(x+.28, start-j*spacing, line, ha="left", va="top",
                fontsize=fontsize, color="#202A30")


def _arrow(ax, start, end, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=19, linewidth=1.55, color=color))


def _save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    paths=[]
    for extension in ("png", "pdf"):
        path=out/f"{stem}.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def overview(out, filter_config):
    sigmas=filter_config["measurement_sigma_xyz"]
    fig, ax = _canvas("第六种方法详解：九维飞行状态 + 物理反弹 + GP 残差",
        "外部仅输入采样时刻与 XYZ；速度和有效升力由轨迹估计。所有量使用测量坐标系，+z 向上。")
    positions=[(.18,10.90),(10.22,10.90),(.18,7.30),(10.22,7.30),
               (.18,3.70),(10.22,3.70),(.18,.10),(10.22,.10)]
    w,h=9.58,3.26
    boxes=[
        ("1  已收到的观测与九维状态", [
            r"$y_i=p(t_i)+\epsilon_i,\quad\epsilon_i\sim\mathcal{N}(0,R_0)$",
            r"$s=[p^T,v^T,b^T]^T\in\mathbb{R}^{9}$",
            rf"$R_0=\operatorname{{diag}}({sigmas[0]:g}^2,{sigmas[1]:g}^2,{sigmas[2]:g}^2)\ \mathrm{{m}}^2$",
            "p：位置 m；v：速度 m/s；b：有效 Magnus 系数 s⁻¹",
            "b 不是测得转速；不得把估计 b 直接解释为真实 RPM"], BLUE),
        ("2  飞行模型与状态转移", [
            r"$\dot p=v,\quad\dot v=g-k_D\|v\|v+b\times v,\quad\dot b=0$",
            r"$g=(0,0,-9.81)^T\ \mathrm{m/s^2},\quad k_D=0.0204321\ \mathrm{m^{-1}}$",
            r"$s(t+\Delta t)=\Phi_{\Delta t}(s(t))$",
            "采用 RK4 积分，最大子步 4 ms；均值 b 在飞行段保持常数",
            "在线协方差允许 b 的过程噪声，表达未建模变化"], GREEN),
        ("3  鲁棒初态拟合（前缀批量路径）", [
            r"$r_{ia}=\{\Phi_{t_i-t_0}(s_0)_a-y_{ia}\}/\sigma_a$",
            r"$\hat s_0=\arg\min_{s_0}\left[\frac{1}{2}\sum_{i,a}\rho(r_{ia}^2)+\frac{\|b_0\|^2}{2\sigma_b^2}\right]$",
            r"$\rho(z)=2(\sqrt{1+z}-1),\quad\sigma_b=0.0414906\ \mathrm{s^{-1}}$",
            "soft-L1 只作用于位置项；b 的高斯先验保持二次损失",
            "只使用已经收到的点；数据不足时返回等待，不虚构速度"], BLUE),
        ("4  首次触地由高度事件决定", [
            r"$q(s)=p_z-h_{\rm eff},\quad h_{\rm eff}=0.0141227\ \mathrm{m}$",
            r"$q(s^-)>0,\quad q(s_{\rm end})\leq0,\quad v_{z,\rm end}<0$",
            r"$\hat t_b=\inf\{t:q(s(t))\leq0,\ v_z(t)<0\}$",
            "子步内对高度过零二分 14 次；应用首次碰撞后锁定事件标志",
            "参考触地时刻只供评分；事件高度是有效平面，并非实测球半径"], BLUE),
        ("5  局部碰撞坐标与五维特征", [
            r"$u=\|v^-_{xy}\|,\ e_t=(v_x^-,v_y^-,0)^T/u$",
            r"$e_n=(0,0,1)^T,\quad e_{\perp}=e_n\times e_t$",
            r"$R=[e_t,e_{\perp},e_n]^T,\quad b^-_{\rm loc}=Rb^-$",
            r"$X=[u,-v_z^-,b_t^-,b_{\perp}^-,b_n^-]\in\mathbb{R}^{5}$",
            "u < 10⁻⁶ m/s 时方向退化，回退到固定物理碰撞"], ORANGE),
        ("6  物理出射速度 + GP 三维修正", [
            r"$v^+_{\rm phys}=D v^-,\quad D=\operatorname{diag}(\alpha_t,\alpha_t,-e_n)$",
            r"$\alpha_t=0.5576036,\quad e_n=0.8339975$",
            r"$v^+=Dv^-+R^T f_{\rm GP}(X),\quad b^+=b^-$",
            r"$p^+_{xy}=p^-_{xy},\quad p_z^+=h_{\rm eff}$",
            "GP 输出是局部 Δv⁺，单位 m/s；没有隐藏速度截断"], ORANGE),
        ("7  GP 训练、标准化与冻结推理", [
            r"$\tilde X=(X-\mu_X)/\sigma_X,\quad\tilde Y=(Y-\mu_Y)/\sigma_Y$",
            r"$k(x_i,x_j)=\exp[-\frac{1}{2}\sum_d(x_{id}-x_{jd})^2/\ell_d^2]$",
            r"$A=K+\operatorname{diag}(\sigma_n^2/w_i+10^{-10})$",
            r"$f_{\rm GP}(X_*)=\mu_Y+\sigma_Y\odot[k_*^T A^{-1}\tilde Y]$",
            "ℓd = 1；σn = 0.6；每个独立碰撞的所有变体总权重为 1"], PURPLE),
        ("8  未来投影与在线状态更新", [
            r"$\hat p(t+H\mid\mathcal{D}_{\leq t})=[\Phi_H^{\rm hybrid}(\hat s_t)]_{p}$",
            r"$r_t=y_t-H_p\hat s_t^-,\quad\hat s_t^+=\hat s_t^-+K_t r_t$",
            "投影器复制状态；未来预测不推进接收时钟或改变历史结果",
            "在线更新 p/v/b 与协方差；GP 权重和物理参数始终保持冻结",
            "100 ms 冻结及恢复是独立策略实验，需与持续在线同口径比较"], GREEN),
    ]
    for (x,y),(title,lines,color) in zip(positions,boxes):
        _box(ax,x,y,w,h,title,lines,color,fontsize=13.4)
    for row in range(4):
        y=positions[2*row][1]+h/2
        _arrow(ax,(9.81,y),(10.14,y))
        if row < 3:
            _arrow(ax,(15.0,positions[2*row+1][1]-.03),
                      (5.0,positions[2*row+2][1]+h+.035))
    fig.text(.03,.026,"边界：训练监督可用训练轨迹的反弹后数据；部署输入仅用当前收到的位置。有效升力与协方差均非已验证真实旋转/置信区间。",
             fontsize=11.5,color=GRAY)
    return _save(fig,out,"gp_method6_detailed_overview")


def gp_formula_detail(out, filter_config):
    q_b=float(filter_config["magnus_drive_sigma"])
    initial_b=float(filter_config["initial_magnus_sigma"])
    q_a=float(filter_config["process_accel_sigma"])
    bounce_p=float(filter_config["bounce_position_sigma"])
    bounce_v=float(filter_config["bounce_velocity_sigma"])
    gate=float(filter_config["gate_threshold"])
    soft_gate=float(filter_config["soft_gate_threshold"])
    recovery_gain=float(filter_config["recovery_max_velocity_gain"])
    fig,ax=_canvas("第六种方法的关键公式与实现边界",
        "所有公式与现有 collision_prefix / identified_dynamics / collision_mapping / identified_filter 代码一致。",
        rows=3,height=16)
    w,h=9.58,3.40
    _box(ax,.18,7.15,w,h,"A  飞行雅可比与协方差传播",[
        r"$F_{c,pv}=I,\quad F_{c,vv}=J_v,\quad F_{c,vb}=-[v]_{\times}$",
        r"$J_v=-k_D(\|v\|I+vv^T/\|v\|)+[b]_{\times}$",
        r"$F=\partial\Phi_{\Delta t}/\partial s,\quad P^-=FP^+F^T+Q(\Delta t)$",
        r"$Q_{p_ap_a}=\sigma_a^2\Delta t^3/3,\ Q_{p_av_a}=\sigma_a^2\Delta t^2/2,\ Q_{v_av_a}=\sigma_a^2\Delta t$",
        rf"$Q_{{bb}}=({q_b:.8f})^2\Delta t\,I,\quad \sigma_a={q_a:g}$（冻结配置过程噪声尺度）",
        "[u]× w = u × w；当前事件传播用 reset Jacobian，未加入完整盐跃项"],GREEN,fontsize=13.2)
    _box(ax,10.22,7.15,w,h,"B  鲁棒拟合与初态不确定性",[
        r"$\rho(z)=2(\sqrt{1+z}-1),\quad \rho'(z)=(1+z)^{-1/2}$",
        r"$C_0\approx(J_{\rm robust}^T J_{\rm robust})^{\dagger}$",
        r"$\hat s_c=\Phi_{t_c-t_0}(\hat s_0),\quad P_c=F C_0 F^T$",
        "测量标准差：[6, 6, 8] mm；soft-L1 标准化尺度为 1",
        "b 分量边界 ±2 s⁻¹；初速度分量边界 ±150 m/s",
        "这是局部线性近似；协方差概率覆盖率尚未校准"],BLUE,fontsize=13.2)
    _box(ax,.18,3.58,w,h,"C  GP 的监督目标、归一化与事件权重",[
        r"$Y_i=R_i(v^+_{i,\rm label}-D v^-_i)$",
        r"$w_i=1/n_{c(i)},\quad\sum_{i:c(i)=c}w_i=1$",
        r"$\mu_X={\sum_i w_i X_i}/{\sum_iw_i}$",
        r"$\sigma_{X,d}^2={\sum_i w_i(X_{id}-\mu_{X,d})^2}/{\sum_iw_i}$",
        "Y 也用同样的加权均值/尺度标准化；仅由训练数组计算",
        "多个缺失末段变体是同一碰撞，不能当成多个独立事件"],PURPLE,fontsize=13.2)
    _box(ax,10.22,3.58,w,h,"D  GP 均值、方差与物理解释",[
        r"$\beta=A^{-1}\tilde Y,\quad \tilde f_*=k_*^T\beta$",
        r"$V_* =\max(1-k_*^TA^{-1}k_*,0)$",
        r"$\operatorname{std}(f_{*,d})=\sigma_{Y,d}\sqrt{V_*}$",
        "3 个输出共享核矩阵，各自拟合残差列；不是深层神经网络",
        "这个方差是所选核/噪声下的条件残差不确定性",
        "当前在线 EKF 未直接融合 GP 后验方差；不能当作校准置信区间"],PURPLE,fontsize=13.2)
    _box(ax,.18,.02,w,h,"E  位置 EKF：硬门控、软降权与 Joseph 形式",[
        r"$H_p=[I_3\ 0\ 0],\quad r=y-H_p\hat s^-$",
        r"$S_0=H_pP^-H_p^T+R_0,\quad d^2=r^T S_0^{-1}r$",
        rf"$d^2>{gate:g}:\ \mathrm{{reject}};\quad R=R_0\max(1,d^2/{soft_gate:g})$",
        r"$K=P^-H_p^T(H_pP^-H_p^T+R)^{-1},\quad\hat s^+=\hat s^-+Kr$",
        r"$P^+=(I-KH_p)P^-(I-KH_p)^T+KRK^T$",
        "Δv = Kv r；普通位置更新尚无专门速度影响上限，可出现早期峰"],BLUE,fontsize=12.9)
    _box(ax,10.22,.02,w,h,"F  接触重置与因果恢复的边界",[
        r"$s^+=\mathcal{R}(s^-),\quad P^+=D\mathcal{R}\,P^-D\mathcal{R}^T+Q_b$",
        rf"$Q_{{b,pp}}={bounce_p:g}^2I,\quad Q_{{b,vv}}={bounce_v:g}^2I$",
        rf"$Q_{{b,bb}}=0.25\,({initial_b:.8f})^2I$（冻结配置的初始 b 方差）",
        "原反弹后确认：至少 10 点/30 ms，并通过短窗质量检查",
        rf"$\|K_{{vv,\rm recovery}}\|_2\leq{recovery_gain:g}$",
        "速度噪声膨胀实现渐进融合，随后使用 Joseph 协方差更新",
        "此恢复保护不等于普通位置更新限幅；需区分两个通道"],ORANGE,fontsize=13.2)
    fig.text(.03,.028,"在线过程噪声/初始协方差读取本轮 selected_model.json；其它数值保留约 7 位。没有重训 GP、调整物理参数或改变滤波器。",
             fontsize=11.5,color=GRAY)
    return _save(fig,out,"gp_method6_formulas_detail")


def causal_protocol(out, config, filter_config):
    protocol = dict(config)
    init = protocol.get("initialization", "80 ms / 至少 8 点的因果初始化；初始化完成前仅显示等待")
    freeze = protocol.get("freeze_trigger", "内部轨迹首次下降穿过 h_eff；不读取参考触地时间")
    hold = protocol.get("hold_duration_s", .1)
    recovery = protocol.get("resume_rule", "当前新帧普通更新一次；历史缓存允许因果恢复，不逐帧重放")
    initial_b=float(filter_config["initial_magnus_sigma"])
    initial_v=float(filter_config["initial_velocity_sigma"])
    fig,ax=_canvas("从轨迹起点开始：高度触地 → 冻结 100 ms → 恢复在线",
        "比较各策略的真实未来预测误差；首点没有可识别的速度，预测从因果初始化完成后开始。",
        rows=3,height=15)
    _box(ax,.18,7.12,19.56,3.27,"1  从数据开始接收，先建立可识别状态",[
        init,
        rf"自然首段只拟合 6D p/v，soft-L1 尺度 1.5；b 均值 0，$\sigma_b={initial_b:.8f}\ \mathrm{{s^{{-1}}}}$，$\sigma_v={initial_v:g}\ \mathrm{{m/s}}$",
        r"$\mathcal{D}_t=\{(t_i,y_i):t_i\leq t\},\quad \hat s_t=\mathcal{E}(\mathcal{D}_t)$",
        "所有比较使用相同的采样时间、首段初始化和冻结权重；参考事件标签只用于事后评分",
        "只用位置时，一个点无法同时决定速度与有效升力。起始等待区应单独标出，不能补成零误差"],BLUE,fontsize=14.5)
    _box(ax,.18,3.55,9.58,3.30,"2  模型高度事件触发一次冻结",[
        freeze,
        r"$\hat t_b=\inf\{t:\hat p_z(t)\leq h_{\rm eff},\ \hat v_z(t)<0\}$",
        rf"$\hat t_b\leq t<\hat t_b+{float(hold):.3f}:\quad\hat s_t=\Phi^{{\rm hybrid}}_{{t-\hat t_b}}(\hat s_{{\hat t_b}})$",
        r"$K_t=0$；冻结期间仍执行动力学、事件重置和协方差传播",
        "当前帧先检查 guard 再测量更新；冻结帧只记 recent，不融合"],ORANGE,fontsize=13.8)
    _box(ax,10.22,3.55,9.58,3.30,"3  恢复更新，检验重新接入的暂态",[
        recovery,
        rf"$t\geq\hat t_b+{float(hold):.3f}:\quad \hat s_t^+=\hat s_t^-+K_t r_t$",
        "比较原持续在线 / 高度在线 / 触地永久冻结 / 冻结100 ms恢复",
        "冻结策略可能把误差峰延后到恢复时；恢复附近需要密集评分",
        "高度前只延后反弹确认重置；正常触地前再捕获保留"],GREEN,fontsize=13.8)
    _box(ax,.18,.02,19.56,3.27,"4  误差有两种口径；最优组合必须分阶段判断",[
        r"$e_{j,t,H}=\|\hat p_j(t+H\mid\mathcal{D}_{\leq t})-p^{\rm ref}_j(t+H)\|_2$",
        r"$E_{j,B,H}=\sqrt{\operatorname{mean}_{t\in B}(e_{j,t,H}^2)},\quad\operatorname{RMSE}_{B,H}=\sqrt{N_B^{-1}\sum_j E_{j,B,H}^2}$",
        "始终向前 H：每次均预测未来 100/200/500 ms；固定目标：观察同一未来时刻估计如何逐步收敛",
        "当前状态拟合误差不是未来预测误差；整体最优同时看飞行、接触 0–100 ms、恢复峰与后续误差",
        "N 为独立轨迹数；P95 是每条轨迹桶内 RMS 的经验分位数。训练/验证选型，历史测试仅回归"],PURPLE,fontsize=14.0)
    _arrow(ax,(10,7.08),(10,6.95))
    _arrow(ax,(9.81,5.15),(10.14,5.15))
    _arrow(ax,(10,3.50),(10,3.37))
    fig.text(.03,.047,"另 init_hold100：80 ms 初始化后即冻结，直到内部高度事件 +100 ms 恢复；freeze_after_init：初始化后永久冻结。",
             fontsize=11.5,color=GRAY)
    fig.text(.03,.022,"这些分支都不继续融合触地前新点。全部高度策略无参考触地输入；是否最优以训练/验证选型和同口径实测为准。",
             fontsize=11.5,color=GRAY)
    return _save(fig,out,"gp_height_hold_protocol_explained")


def render(out, protocol=None, selected_model=None):
    selected_model=(Path(__file__).resolve().parents[1]/"artifacts/tennis_identified/20261009_training_prior/selected_model.json"
        if selected_model is None else Path(selected_model))
    model_bytes=selected_model.read_bytes()
    model=json.loads(model_bytes)
    filter_config=model["filter"]
    required=("magnus_drive_sigma","initial_magnus_sigma","initial_velocity_sigma",
        "process_accel_sigma","measurement_sigma_xyz","bounce_position_sigma","bounce_velocity_sigma")
    if any(k not in filter_config for k in required):
        raise ValueError("Selected frozen model missing actual filter parameters")
    font = _style()
    files = overview(out, filter_config)+gp_formula_detail(out, filter_config)
    if protocol is not None:
        files += causal_protocol(out, protocol, filter_config)
    return {"schema":"GP_METHOD6_DETAILED_DIAGRAMS/1", "font":font,
        "files":files, "formal_platform":"Ubuntu project .venv-tennis",
        "prediction_or_fit_performed":False, "protocol":protocol,
        "actual_selected_model":str(selected_model),"actual_selected_model_sha256":hashlib.sha256(model_bytes).hexdigest(),
        "actual_filter_config":filter_config,
        "natural_initialization":"6D p/v soft-L1 f_scale1.5; mean b=0; configured initial b covariance",
        "batch_initialization":"9D p/v/b soft-L1 position f_scale1 plus independent Gaussian b prior"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol-json",type=Path,
        help="Explicit causal protocol description; omit to render formula diagrams only")
    parser.add_argument("--selected-model",type=Path,
        help="Frozen selected_model.json containing this run's actual filter configuration")
    args=parser.parse_args()
    protocol=None if args.protocol_json is None else json.loads(args.protocol_json.read_text())
    result=render(args.output, protocol, args.selected_model)
    (args.output/"detailed_method_diagrams.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
