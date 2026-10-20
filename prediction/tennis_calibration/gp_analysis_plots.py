"""Scientific figures for method 6, without fitting or modifying its checkpoint.

This module only scores supplied predictions. Frozen targets share the reference
event clock; online comparison uses each row's own ``frozen_xyz`` at the exact
same absolute target time. Irregular raw capture timestamps are grouped in
20-ms bins, giving each independent trajectory one vote within a bin. Formal
evaluation and rendering are performed in the Ubuntu project environment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

BLUE = "#167DAB"
ORANGE = "#D47624"
PURPLE = "#7A5797"
GRAY = "#616C76"
GREEN = "#4C8D6C"


def _style():
    names = {item.name for item in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "Arial Unicode MS") if name in names]
    matplotlib.rcParams.update({"font.family": preferred+["DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 9, "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .18, "pdf.fonttype": 42, "ps.fonttype": 42})
    return preferred[0] if preferred else "DejaVu Sans (Chinese font missing)"


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _xyz(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return array if array.shape == (3,) and np.all(np.isfinite(array)) else None


def _valid(row):
    # Existing labels use ``valid`` for label validity, so prediction finiteness
    # and a prediction failure reason must also be checked independently.
    flag = row.get("valid", True)
    if isinstance(flag, str):
        flag = flag.lower() not in ("false", "0", "no", "")
    return bool(flag) and not bool(row.get("prediction_reason"))


def _label_valid(row):
    flag = row.get("valid", True)
    return flag.lower() not in ("false", "0", "no", "") if isinstance(flag,str) else bool(flag)


def _key(row, protocol, stride):
    if protocol == "frozen":
        return (round(_number(row.get("tau_s")), 8),)
    relative = _number(row.get("relative_capture_s"))
    capture_bin = round(round(relative/stride)*stride, 8) if np.isfinite(relative) else np.nan
    h = row.get("tau_s") if protocol == "target" else row.get("horizon_s")
    return (round(_number(h), 8), capture_bin)


def aggregate_gp_rows(rows, protocol, stride=.02, prediction_field="predicted_xyz", paired=False):
    """Equal-trajectory RMSE/P95 and explicit label/prediction coverage per bin.

    When ``paired`` is true, both the online and frozen predictions must succeed
    in a row; this makes their curves use identical targets. P95 is the empirical
    95th percentile of trajectory RMS errors in that bin, not GP uncertainty.
    """
    groups = defaultdict(list)
    invalid_keys = 0
    for row in rows:
        key = _key(row, protocol, stride)
        if not all(np.isfinite(value) for value in key):
            invalid_keys += 1
            continue
        groups[key].append(row)
    cells = []
    for key, members in sorted(groups.items()):
        ids = {str(row["trajectory_id"]) for row in members}
        labeled, scored = set(), set()
        squares = defaultdict(list)
        axis_squares = defaultdict(list)
        reasons = Counter()
        for row in members:
            identity = str(row["trajectory_id"])
            target = _xyz(row.get("target_xyz"))
            prediction = _xyz(row.get(prediction_field))
            if target is None or not _label_valid(row):
                reasons[str(row.get("reason", "missing_reference_label"))] += 1
                continue
            labeled.add(identity)
            good = prediction is not None
            if prediction_field == "predicted_xyz":
                good = good and _valid(row)
            else:
                good = good and not bool(row.get("frozen_reason"))
            if paired:
                good = good and _xyz(row.get("predicted_xyz")) is not None and _valid(row)
                good = good and _xyz(row.get("frozen_xyz")) is not None and not bool(row.get("frozen_reason"))
            if not good:
                reasons[str(row.get("prediction_reason") or row.get("frozen_reason") or "prediction_missing_or_nonfinite")] += 1
                continue
            scored.add(identity)
            delta = prediction-target
            squares[identity].append(float(delta@delta))
            axis_squares[identity].append(delta*delta)
        trajectory_mse = np.asarray([np.mean(values) for values in squares.values()])
        trajectory_rms = np.sqrt(trajectory_mse)
        axis = np.asarray([np.mean(axis_squares[identity], axis=0) for identity in squares])
        cell = {"key": list(key), "requested_trajectories": len(ids),
            "labeled_trajectories": len(labeled), "successful_trajectories": len(scored),
            "requested_rows": len(members), "successful_rows": sum(len(values) for values in squares.values()),
            "label_coverage": len(labeled)/len(ids) if ids else None,
            "prediction_coverage_given_label": len(scored)/len(labeled) if labeled else None,
            "failures_or_missing_labels": dict(reasons),
            "rmse_m": float(np.sqrt(np.mean(trajectory_mse))) if len(trajectory_mse) else None,
            "p50_m": float(np.quantile(trajectory_rms,.5)) if len(trajectory_rms) else None,
            "p95_m": float(np.quantile(trajectory_rms,.95)) if len(trajectory_rms) else None,
            "axis_rmse_m": np.sqrt(np.mean(axis,axis=0)).tolist() if len(axis) else None}
        if protocol == "frozen":
            cell["tau_s"] = key[0]
        else:
            cell["target_tau_s" if protocol == "target" else "horizon_s"] = key[0]
            cell["relative_capture_bin_s"] = key[1]
        cells.append(cell)
    return {"cells": cells, "invalid_group_key_rows": invalid_keys,
        "definition": "sqrt(mean_trajectory(mean_within_bin(||predicted_xyz-reference_xyz||^2)))",
        "p95_definition": "empirical percentile of trajectory RMS errors in each bin",
        "capture_bin_width_s": None if protocol == "frozen" else stride,
        "paired": paired, "prediction_field": prediction_field}


def _save(fig, output, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = output/(stem+"."+extension)
        fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _note(fig, text):
    fig.text(.02,.012,text,ha="left",va="bottom",fontsize=8,color=GRAY)


def _values(cells, field, scale=1.):
    return np.asarray([_number(cell.get(field))*scale for cell in cells])


def _curve(ax, cells, color=BLUE, label="GP 第六方案：RMSE", linestyle="-", p95=False):
    x = _values(cells,"relative_capture_bin_s")
    ax.plot(x,_values(cells,"rmse_m",100),color=color,label=label,lw=2,ls=linestyle)
    if p95:
        ax.plot(x,_values(cells,"p95_m",100),color=color,lw=1.2,ls=":",label="在线 P95")


def _coverage(ax, cells, xfield):
    x = _values(cells,xfield)
    ax.step(x,_values(cells,"requested_trajectories"),where="mid",color="#B6BCC2",label="请求轨迹数",lw=1.2)
    ax.step(x,_values(cells,"labeled_trajectories"),where="mid",color=GRAY,label="有参考标签 N",lw=1.6)
    ax.step(x,_values(cells,"successful_trajectories"),where="mid",color=GREEN,label="成功预测 N",lw=1.2,ls="--")
    ax.set_ylabel("独立轨迹 N")
    ax.set_ylim(bottom=0)


def _method_figure(output, summary):
    fig,ax=plt.subplots(figsize=(15,10.8))
    ax.set(xlim=(0,15),ylim=(0,10.8));ax.axis("off")
    fig.suptitle("第六种方法：有效物理反弹 + GP 出射速度残差",fontsize=19,y=.985)
    def box(x,y,w,h,title,text,color=BLUE):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.13",
                                   facecolor=color+"10",edgecolor=color,lw=1.4))
        ax.text(x+.15,y+h-.20,title,fontsize=12,weight="bold",va="top",color=color)
        ax.text(x+.15,y+h-.60,text,fontsize=10,va="top",linespacing=1.65)
    def arrow(start,end,color=GRAY):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,color=color,lw=1.4))
    box(.25,7.35,3.3,2.35,"1  触地前原始观测",
        "只有时间 t、位置 XYZ\n本次采用全部已收到触地前点\nwindow = 0；全部点参与拟合\n不读取触地后观测；不做 72 点抽样")
    box(4.1,7.35,4.55,2.35,"2  鲁棒九维飞行状态估计",
        "状态：位置 p、速度 v、有效升力 b（各 3 维）\n位置误差 soft-L1 + 独立高斯 b 先验\n先验 σb ≈ 0.04149 s⁻¹\n有效 b 从轨迹估计，不能解释为实测 RPM")
    box(9.2,7.35,5.4,2.35,"3  自主预测首次触地事件",
        "从估计状态积分，搜索有效事件平面\n输出预测触地时间、位置、入射 v⁻ / b⁻\n有效高度 h ≈ 0.01412 m\n参考触地时间只用于实验截断与评分")
    arrow((3.65,8.52),(3.98,8.52));arrow((8.75,8.52),(9.08,8.52))
    box(.25,3.98,4.85,2.8,"4  物理映射 + GP 修正", 
        "物理基线：v⁺xy = 0.5576 v⁻xy；v⁺z = −0.8340 v⁻z\n局部框架 R：水平入射切向、横向、法向\nGP 输入 5 维：[|v⁻xy|, −v⁻z, bt, b⊥, bn]\nGP 输出 3 维：局部出射速度残差 Δv⁺\n最终 v⁺ = v⁺physics + Rᵀ Δv⁺；b⁺ = b⁻",ORANGE)
    box(5.7,3.98,4.05,2.8,"5  反弹后飞行积分",
        "dp/dt = v\ndv/dt = g − kD |v|v + b × v\ndb/dt = 0\nkD ≈ 0.02043 m⁻¹\n输出目标时刻的 p、v、b（九维）",GREEN)
    box(10.35,3.98,4.25,2.8,"GP 权重保持冻结",
        "训练样本独立碰撞总权重均为 1\n特征/残差标准化只用训练数据\nRBF 核：长度尺度 1.0；噪声 0.6\n在线更新的是状态及其协方差\n本次没有在线重训练 GP",PURPLE)
    arrow((11.8,7.25),(2.7,6.95));arrow((5.2,5.4),(5.58,5.4))
    box(.25,.68,6.9,2.58,"A  固定预测：以后不再读取观测",
        "全部触地前观测 → 一次估计 → GP 出射状态冻结\n自此只积分飞行；绘制误差随反弹后时间 τ 变化\n所有轨迹在 t参考触地 + τ 评分（不按预测触地移轴）\n时间/触地点/出射速度误差共同进入未来位置误差",ORANGE)
    box(7.7,.68,6.9,2.58,"B  在线预测：收到新点就更新状态",
        "用 A 的同一全部触地前结果初始化 GP-aware EKF\n新 XYZ 观测按时间顺序更新 p、v、b 和协方差\n每次从该时刻的状态副本预测未来 100 / 200 / 500 ms\n评分是未来位置误差，不是当前观测拟合误差",BLUE)
    arrow((7.65,3.85),(3.7,3.45),ORANGE);arrow((7.85,3.85),(11.2,3.45),BLUE)
    _note(fig,"全前缀评估改变了原模型训练时最近 0.4 s 的上游输入分布；本次没有重训。仅首次球地反弹；历史测试回归，非新盲测。")
    fig.subplots_adjust(top=.94,bottom=.06,left=.015,right=.985)
    return _save(fig,output,"gp_method6_explained")


def _frozen_figure(output, metrics, summary):
    cells=metrics["cells"]
    fig,axes=plt.subplots(2,1,figsize=(11.6,7.6),sharex=True,gridspec_kw={"height_ratios":[3,1]})
    x=_values(cells,"tau_s")
    axes[0].plot(x,_values(cells,"rmse_m",100),lw=2.4,color=ORANGE,label="冻结触地前结果：三维位置 RMSE")
    axes[0].plot(x,_values(cells,"p95_m",100),lw=1.8,color=PURPLE,ls="--",label="逐轨迹误差 P95")
    for axis in axes:
        axis.axvspan(0,.5,color=ORANGE,alpha=.055)
        axis.axvline(.5,color=GRAY,ls=":",lw=1)
        axis.set_xlim(0,1.0)
    half_requested=max((cell["requested_trajectories"] for cell in cells),default=0)/2
    reduced=[cell for cell in cells if cell["labeled_trajectories"]<half_requested]
    if reduced:
        first_reduced=reduced[0]["tau_s"]
        for axis in axes:
            axis.axvspan(first_reduced,1.,color=GRAY,alpha=.07)
        axes[0].text(.985,.045,"参考标签低于半数\n回落受样本变化影响",transform=axes[0].transAxes,
                     va="bottom",ha="right",fontsize=9,color=GRAY)
    axes[0].set_ylabel("未来三维位置误差（cm）")
    axes[0].set_title("一次冻结出射状态后的误差累积：使用触地前所有原始观测")
    axes[0].legend(loc="upper left")
    _coverage(axes[1],cells,"tau_s")
    axes[1].set_xlabel("参考首次触地后的飞行时间 τ（s）")
    axes[1].legend(loc="upper right",ncol=3,fontsize=8)
    _note(fig,"浅色区为主要报告范围 0–0.5 s；长时曲线需结合下方 N 阅读。失败和缺标签保留在 N。\nRMSE = √mean(||预测 XYZ − 测量参考 XYZ||²)，每条独立轨迹等权。P95 为经验误差尾部，不是置信区间。")
    fig.subplots_adjust(hspace=.10,bottom=.11,top=.93,left=.10,right=.97)
    return _save(fig,output,"gp_frozen_error_vs_flight_time")


def _online_figure(output, online, frozen, paired_online, summary, stem="gp_online_future_error", fullstream=False):
    # Preserve the existing rendering API and saved comparison statistics; the
    # online display now shows only online RMSE/P95 and its coverage.
    horizons=sorted({cell["horizon_s"] for cell in online["cells"]})
    fig,axes=plt.subplots(2,len(horizons),figsize=(5.1*len(horizons),7.4),squeeze=False,sharex="col",
                          gridspec_kw={"height_ratios":[3,1]})
    for column,h in enumerate(horizons):
        cells=[cell for cell in online["cells"] if cell["horizon_s"]==h]
        _curve(axes[0,column],cells,p95=True,label="在线：未来位置 RMSE")
        axes[0,column].set_title(f"每次预测未来 {h*1000:g} ms")
        axes[0,column].set_ylabel("未来三维位置误差（cm）")
        axes[0,column].legend(loc="upper left",fontsize=8)
        axes[0,column].set_ylim(bottom=0)
        _coverage(axes[1,column],cells,"relative_capture_bin_s")
        axes[1,column].set_xlabel("收到观测的时刻 − 参考触地时刻（s）")
        for axis in axes[:,column]:
            axis.axvline(0,color=GRAY,ls=":",lw=1)
    axes[1,-1].legend(loc="best",fontsize=7)
    fig.suptitle("在线状态更新对未来位置预测的影响"+("：从初始飞行开始" if fullstream else "：同一全部触地前结果初始化"),fontsize=15)
    _note(fig,"每条轨迹在 20 ms 观测时刻网格中等权（±10 ms）；评分目标仍使用实际 capture + horizon 的绝对时钟。\n在线曲线不使用未来观测；横轴推进时目标也向未来平移，因此不是同一个目标逐渐变准。P95 为经验误差尾部；历史测试回归。")
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.11,wspace=.28,left=.06,right=.98)
    return _save(fig,output,stem)


def _target_figure(output, online, frozen, summary, stem="gp_fixed_target_refinement", fullstream=False):
    targets=sorted({cell["target_tau_s"] for cell in online["cells"]})
    fig,axes=plt.subplots(2,len(targets),figsize=(5.1*len(targets),7.4),squeeze=False,sharex="col",
                          gridspec_kw={"height_ratios":[3,1]})
    for column,h in enumerate(targets):
        cells=[cell for cell in online["cells"] if cell["target_tau_s"]==h]
        _curve(axes[0,column],cells,p95=True,label="在线：固定目标位置 RMSE")
        axes[0,column].set_title(f"固定目标：参考触地后 {h*1000:g} ms")
        axes[0,column].set_ylabel("固定未来目标的三维误差（cm）")
        axes[0,column].legend(loc="upper left",fontsize=8)
        axes[0,column].set_ylim(bottom=0)
        _coverage(axes[1,column],cells,"relative_capture_bin_s")
        axes[1,column].set_xlabel("观测更新时刻 − 参考触地时刻（s）")
        for axis in axes[:,column]:
            axis.axvline(0,color=GRAY,ls=":",lw=1)
            axis.axvline(h,color=PURPLE,ls=":",lw=1)
    axes[1,-1].legend(loc="best",fontsize=7)
    fig.suptitle("同一个未来目标随新观测逐渐修正"+("：从初始飞行开始" if fullstream else "：同一全部触地前结果初始化"),fontsize=15)
    _note(fig,"每条轨迹的目标绝对时间固定为 t参考触地 + 100 / 200 / 500 ms；随着更新，剩余预测时距缩短（至少 20 ms）。\n参考触地时刻只定义离线评分目标，不能作为在线预测器输入。曲线按 20 ms 网格等轨迹汇总；历史测试回归。")
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.11,wspace=.28,left=.06,right=.98)
    return _save(fig,output,stem)


def _example_figure(output, example):
    if not example:
        return []
    tb=_number(example.get("reference_tb"))
    scope_end=_number(example.get("scope_end_s",example.get("eval_scope_end_s")))
    if not np.isfinite(scope_end):
        scope_end=tb+1.
    events=example.get("reference_events",example.get("references_events",
           example.get("reference_event_times_s",example.get("bounce_event_times_s",[tb]))))
    events=[float(item.get("time_s")) if isinstance(item,dict) else float(item) for item in events]
    reference_times=np.asarray(example.get("reference_times",[]),dtype=float)
    reference=np.asarray(example.get("reference_xyz",[]),dtype=float)
    frozen_times=np.asarray(example.get("frozen_times",[]),dtype=float)
    frozen=np.asarray(example.get("frozen_xyz",[]),dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or frozen.ndim!=2 or frozen.shape[1]!=3:
        return []
    forecasts=list(example.get("forecasts",[]))[:5]
    fig=plt.figure(figsize=(15,9.1))
    ax=fig.add_subplot(2,2,1,projection="3d")
    side=fig.add_subplot(2,2,2)
    height=fig.add_subplot(2,2,3)
    errors=fig.add_subplot(2,2,4)
    selected=(reference_times>=tb-.08)&(reference_times<=min(tb+1.,scope_end))
    t=reference_times[selected];xyz=reference[selected]
    in_scope=frozen_times<=scope_end+1e-12
    frozen_times,frozen=frozen_times[in_scope],frozen[in_scope]
    # Break display segments at the same large holes or event brackets that
    # cannot produce interpolation labels. Keep actual retained points intact.
    display_t,display_xyz=[],[]
    for index,(stamp,point) in enumerate(zip(t,xyz)):
        if index and (stamp-t[index-1]>.012 or any(t[index-1]<=event<=stamp for event in events)):
            display_t.append(np.nan);display_xyz.append([np.nan]*3)
        display_t.append(stamp);display_xyz.append(point.tolist())
    display_t=np.asarray(display_t);display_xyz=np.asarray(display_xyz)
    ax.plot(*display_xyz.T,color=GRAY,lw=2,label="测量参考（仅评分/展示）")
    side.plot(display_xyz[:,0],display_xyz[:,2],color=GRAY,lw=2)
    height.plot(display_t-tb,display_xyz[:,2],color=GRAY,lw=2,label="测量参考")
    ax.plot(*frozen.T,color=ORANGE,lw=2,label="触地前冻结预测")
    side.plot(frozen[:,0],frozen[:,2],color=ORANGE,lw=2)
    height.plot(frozen_times-tb,frozen[:,2],color=ORANGE,lw=2,label="冻结预测")
    colors=plt.get_cmap("viridis")(np.linspace(.1,.85,max(len(forecasts),1)))
    # Reuse the evaluator instead of reimplementing its phase/gap conditions.
    # NaN values keep the error curve broken wherever a label is unavailable.
    def visual_error(times,pred):
        from .online_data import future_labels
        if not len(times):
            return np.asarray([]),np.asarray([])
        anchor=float(np.min(times))
        labels=future_labels({"eval_times":reference_times,"eval_xyz":reference,
            "bounce_reference_s":tb,"bounce_event_times_s":events,"eval_scope_end_s":scope_end},
            anchor,times-anchor,max_gap_s=.012,bounce_padding_s=.003)
        errors=np.full(len(times),np.nan)
        for index,label in enumerate(labels):
            if label["valid"]:
                errors[index]=np.linalg.norm(pred[index]-np.asarray(label["target_xyz"]))*100
        return times-tb,errors
    et,ev=visual_error(frozen_times,frozen)
    errors.plot(et,ev,color=ORANGE,lw=2,label="冻结预测")
    for forecast,color in zip(forecasts,colors):
        capture=_number(forecast.get("capture_t_s"))
        ft=np.asarray(forecast.get("times",[]),dtype=float)
        fp=np.asarray(forecast.get("xyz",[]),dtype=float)
        if fp.ndim!=2 or fp.shape!=(len(ft),3):
            continue
        future=(ft>=capture-1e-12)&(ft<=scope_end+1e-12)
        ft,fp=ft[future],fp[future]
        if not len(ft):
            continue
        label=f"更新 {(capture-tb)*1000:+.0f} ms"
        ax.plot(*fp.T,color=color,lw=1.7,label=label)
        side.plot(fp[:,0],fp[:,2],color=color,lw=1.7)
        height.plot(ft-tb,fp[:,2],color=color,lw=1.7,label=label)
        height.scatter([capture-tb],[fp[0,2]],color=color,s=25)
        et,ev=visual_error(ft,fp)
        errors.plot(et,ev,color=color,lw=1.5,label=label)
    ax.set(xlabel="X（m）",ylabel="Y（m）",zlabel="Z（m）",title="真实测量与未来预测扇形")
    ax.view_init(elev=24,azim=-63)
    ax.legend(fontsize=7,loc="upper left")
    side.set(xlabel="X（m）",ylabel="Z（m）",title="侧视轨迹")
    height.set(xlabel="参考触地后的时间（s）",ylabel="高度 Z（m）",title="每次更新只绘制该时刻以后的预测")
    height.axvline(0,color=GRAY,lw=1,ls=":")
    errors.set(xlabel="参考触地后的目标时间（s）",ylabel="三维位置误差（cm）",title="单例误差曲线（采用正式参考插值规则）")
    errors.legend(fontsize=7,ncol=2)
    fig.suptitle("确定性示例："+str(example.get("trajectory_id","")),fontsize=11,y=.98)
    _note(fig,"示例为数据加载顺序中的第一条可用轨迹；未按误差挑选。仅展示第一次反弹范围。超过 12 ms 空洞、触地 ±3 ms 及跨碰撞插值不评分。\n线扇起点之后才是预测；完整测量参考仅用于展示和离线评分。")
    fig.subplots_adjust(top=.92,bottom=.09,wspace=.20,hspace=.28)
    return _save(fig,output,"gp_example_prediction_fan")


def make_gp_analysis_plots(out, frozen_rows, online_rows, target_rows, summary=None, example=None,
                           *, fullstream_rows=None, fulltarget_rows=None):
    """Write Chinese PNG/PDF figures and auditable per-bin metrics.

    Row fields: trajectory_id, tau_s, relative_capture_s, horizon_s,
    target_xyz, predicted_xyz, valid, prediction_reason. Online/fixed-target
    rows optionally contain frozen_xyz/frozen_reason at the identical target.
    Return a serializable dictionary; weights and raw results remain untouched.
    """
    summary=summary or {}
    output=Path(out)/"plots";output.mkdir(parents=True,exist_ok=True)
    font=_style()
    frozen_rows,online_rows,target_rows=list(frozen_rows),list(online_rows),list(target_rows)
    stride=float(summary.get("anchor_stride_s",.02))
    metrics={"schema":"GP_METHOD6_PLOT_METRICS/1", "capture_bin_width_s":stride,
        "capture_bin_rule":"nearest grid point; individual target absolute timestamps unchanged",
        "future_reference":"measurement labels supplied by evaluator; never use plotted MuJoCo state as truth",
        "frozen":aggregate_gp_rows(frozen_rows,"frozen",stride),
        "online":aggregate_gp_rows(online_rows,"online",stride),
        "online_common_success":aggregate_gp_rows(online_rows,"online",stride,paired=True),
        "online_frozen_matched":aggregate_gp_rows(online_rows,"online",stride,"frozen_xyz",paired=True),
        "fixed_target":aggregate_gp_rows(target_rows,"target",stride),
        "fixed_target_frozen_matched":aggregate_gp_rows(target_rows,"target",stride,"frozen_xyz",paired=True)}
    paths=[]
    paths.extend(_method_figure(output,summary))
    if metrics["frozen"]["cells"]:
        paths.extend(_frozen_figure(output,metrics["frozen"],summary))
    if metrics["online"]["cells"]:
        paths.extend(_online_figure(output,metrics["online"],metrics["online_frozen_matched"],metrics["online_common_success"],summary))
    if metrics["fixed_target"]["cells"]:
        paths.extend(_target_figure(output,metrics["fixed_target"],metrics["fixed_target_frozen_matched"],summary))
    paths.extend(_example_figure(output,example))
    if fullstream_rows:
        fullstream_rows=list(fullstream_rows)
        metrics["fullstream"]=aggregate_gp_rows(fullstream_rows,"online",stride)
        metrics["fullstream_common_success"]=aggregate_gp_rows(fullstream_rows,"online",stride,paired=True)
        metrics["fullstream_frozen_matched"]=aggregate_gp_rows(fullstream_rows,"online",stride,"frozen_xyz",paired=True)
        paths.extend(_online_figure(output,metrics["fullstream"],metrics["fullstream_frozen_matched"],metrics["fullstream_common_success"],summary,"gp_fullstream_future_error",True))
    if fulltarget_rows:
        fulltarget_rows=list(fulltarget_rows)
        metrics["fulltarget"]=aggregate_gp_rows(fulltarget_rows,"target",stride)
        metrics["fulltarget_frozen_matched"]=aggregate_gp_rows(fulltarget_rows,"target",stride,"frozen_xyz",paired=True)
        paths.extend(_target_figure(output,metrics["fulltarget"],metrics["fulltarget_frozen_matched"],summary,"gp_fullstream_target_refinement",True))
    metric_path=Path(out)/"plot_metrics.json"
    metric_path.write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return {"paths":paths,"metrics_json":str(metric_path),"font":font,
        "rmse_definition":metrics["frozen"]["definition"],"capture_bin_width_s":stride,
        "warning":"Historical exposed test is regression; full prefix changes old recent-0.4-s input distribution; GP is not retrained"}
