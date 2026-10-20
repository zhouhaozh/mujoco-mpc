"""Read-only figures for the causal height/hold policy comparison.

Reads the completed summary, never predictions or reference-event inputs.  The
six primary candidates are selected on validation only.  Formal rendering is
in the Ubuntu project .venv-tennis; no fitting or metric recomputation occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .gp_analysis_plots import _style


ROOT=Path(__file__).resolve().parents[1]
POLICIES=("continuous","height_online","hold100","permanent_hold",
          "init_hold100","freeze_after_init")
HORIZONS=(0.,.1,.2,.5)
COLORS=("#25699D","#268775","#7148A1","#647383","#9B3E62","#2C9AAA")
POLICY_NAMES={
    "continuous":"持续在线（原逻辑）",
    "height_online":"高度事件在线",
    "hold100":"高度触地冻结100ms后在线",
    "permanent_hold":"高度触地后永久冻结",
    "init_hold100":"初始化冻结至触地+100ms后在线",
    "freeze_after_init":"初始化后永久预测",
}
MODELS={"fixed":"固定物理","ridge1":"一次 Ridge","ridge2":"二次 Ridge",
        "gp":"GP 残差","legacy_ekf6":"旧 EKF6","legacy_ekf6_mlp":"旧 EKF6 + MLP"}


def _n(row):
    return int(row.get("N",row.get("successful_trajectories",0)))


def _float(value):
    return float(value) if value is not None and np.isfinite(float(value)) else np.nan


def _cells(summary,split,candidate,axis,h):
    return sorted((c for c in summary["curves"] if c["split"]==split
        and c["candidate"]==candidate and c["axis"]==axis
        and abs(float(c["horizon_s"])-h)<1e-9),key=lambda c:c["time_bin_s"])


def _candidate_label(candidate,policy=None):
    if ":" in candidate:
        model,p=candidate.split(":",1)
        return f"{POLICY_NAMES.get(policy or p,p)} · {MODELS.get(model,model)}"
    return MODELS.get(candidate,candidate)


def _selection(summary):
    selection=summary.get("validation_only_selection",{})
    per=selection.get("per_policy",{})
    missing=[p for p in POLICIES if p not in per]
    if missing:
        raise ValueError(f"Validation selection missing policies: {missing}")
    return [(per[p],_candidate_label(per[p],p),COLORS[i]) for i,p in enumerate(POLICIES)]


def _grid_series(cells,axis,metric,limits=None):
    if limits is not None:
        cells=[c for c in cells if limits[0]-1e-10<=c["time_bin_s"]<=limits[1]+1e-10]
    if not cells:
        return np.array([]),np.array([]),np.array([]),[]
    step=.0025 if axis=="contact_dense2p5" else .02
    lookup={round(float(c["time_bin_s"]),8):c for c in cells}
    lo,hi=min(lookup),max(lookup)
    grid=np.arange(round(lo/step),round(hi/step)+1,dtype=int)*step
    values=np.asarray([_float(lookup.get(round(float(t),8),{}).get(metric))*100 for t in grid])
    count=np.asarray([_n(lookup[round(float(t),8)]) if round(float(t),8) in lookup else np.nan for t in grid])
    return grid,values,count,cells


def _zoom_if_needed(ax,series):
    values=np.concatenate([y[np.isfinite(y)] for _,y,_,_ in series]) if series else np.array([])
    if len(values)<10:
        return None
    typical=[float(np.quantile(y[np.isfinite(y)],.85)) for _,y,_,_ in series if np.isfinite(y).any()]
    q=float(np.median(typical));peak=float(np.max(values))
    if q<=0 or peak/q<5:
        return None
    inset=ax.inset_axes([.57,.55,.40,.37])
    for x,y,color,_ in series:
        inset.plot(x,y,color=color,lw=1.05,alpha=.9)
    inset.set_ylim(0,q*1.15)
    inset.tick_params(labelsize=6,pad=1)
    inset.set_title("低误差区域放大；主轴保留完整峰",fontsize=6.2,pad=2)
    inset.grid(alpha=.16)
    return {"ylim_cm":[0,q*1.15],"purpose":"visual zoom only; full main-axis peak retained"}


def _panel_figure(summary,out,split,candidates,axis,stem,title,metric="rmse_m",limits=None,validation_selected=False):
    fig,axes=plt.subplots(2,4,figsize=(22,8.9),sharex="col",
        gridspec_kw={"height_ratios":[3.4,1.05],"hspace":.12,"wspace":.25})
    fig.subplots_adjust(left=.055,right=.985,top=.80,bottom=.18)
    metric_name="位置 RMSE" if metric=="rmse_m" else "位置 P95"
    fig.suptitle(title+f"｜{metric_name} 与独立轨迹覆盖",fontsize=18,y=.97)
    subtitle=("验证集为每种策略选模型；" if validation_selected else "固定同 GP 策略对照；")
    fig.text(.5,.918,subtitle+"图中显示历史测试回归，不是新盲测。未来预测与当前后验跟踪分开标注。",
        ha="center",fontsize=11,color="#66717A")
    handles=[Line2D([0],[0],color=color,lw=2,label=label) for _,label,color in candidates]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.893),ncol=3,
        fontsize=9.5,frameon=False,handlelength=2.7,columnspacing=1.9)
    panels=[]
    for j,h in enumerate(HORIZONS):
        upper,lower=axes[:,j]
        upper.set_title("H = 0：当前后验跟踪" if h==0 else f"每次预测未来 {h*1000:.0f} ms",fontsize=12,pad=10)
        series=[];curve_details=[];maxn=0
        for candidate,label,color in candidates:
            raw=_cells(summary,split,candidate,axis,h)
            x,y,n,cells=_grid_series(raw,axis,metric,limits)
            if len(x):
                upper.plot(x,y,color=color,lw=1.55,alpha=.93)
                lower.plot(x,n,color=color,lw=1.35,alpha=.83)
                maxn=max(maxn,int(np.nanmax(n)))
            series.append((x,y,color,label))
            curve_details.append({"candidate":candidate,"label":label,"color":color,
                "cells":len(cells),"max_cm":float(np.nanmax(y)) if np.isfinite(y).any() else None,
                "max_successful_N":int(np.nanmax(n)) if np.isfinite(n).any() else 0,
                "points":[{"time_s":float(c["time_bin_s"]),"value_cm":_float(c.get(metric))*100 if c.get(metric) is not None else None,
                           "N":_n(c),"labeled_N":c.get("labeled_trajectories"),
                           "failed_labeled_rows":c.get("failed_labeled_rows",0)} for c in cells]})
        vals=np.concatenate([y[np.isfinite(y)] for _,y,_,_ in series])
        maximum=float(np.max(vals)) if len(vals) else 1.
        upper.set_ylim(0,max(.1,maximum*1.07))
        lower.set_ylim(-.4,maxn+1.5)
        lower.set_yticks(sorted({0,maxn//2,maxn}))
        for ax in (upper,lower):
            if limits is not None:
                ax.set_xlim(limits)
            elif axis=="elapsed20":
                allx=np.concatenate([x for x,_,_,_ in series])
                ax.set_xlim(0,float(np.max(allx)) if len(allx) else 1.)
            else:
                allx=np.concatenate([x for x,_,_,_ in series])
                if len(allx):
                    ax.set_xlim(float(np.min(allx)),float(np.max(allx)))
            if axis=="elapsed20":
                ax.axvspan(0,.08,color="#D8DDE3",alpha=.60,zorder=0)
            else:
                ax.axvline(0,color="#697784",lw=.9,ls=":",alpha=.8)
                ax.axvline(.1,color="#A1AAB2",lw=.9,ls="--",alpha=.8)
            ax.grid(alpha=.20)
            ax.tick_params(labelsize=9)
        inset=_zoom_if_needed(upper,series)
        upper.set_ylabel(f"{metric_name} / cm" if j==0 else "")
        lower.set_ylabel("有效轨迹 N" if j==0 else "")
        lower.set_xlabel("从首个采样开始 / s" if axis=="elapsed20" else "相对参考首次触地 / s")
        panels.append({"horizon_s":h,"ylabel":metric_name,"ylim_cm":list(upper.get_ylim()),
            "xlim_s":list(upper.get_xlim()),"full_visible_max_cm":maximum,"inset":inset,"curves":curve_details})
    if axis=="elapsed20":
        note="灰带：首 80 ms 因果初始化等待；无预测区不补零。不同样本首次触地时刻不同，本横轴不画统一触地/恢复线。"
    else:
        note="虚线 +100 ms 仅是参考触地后的评分位置；每条样本实际恢复由其预测高度事件决定，不能解释为统一恢复时刻。"
    fig.text(.055,.110,note,fontsize=10.2,color="#566470")
    fig.text(.055,.068,"H=0 是后验对位置观测的跟踪误差；H>0 是因果未来预测误差。RMSE/P95 先按每条轨迹桶内 RMS 聚合；下图 N 随真值覆盖及预测成功变化。",
        fontsize=10.2,color="#566470")
    fig.text(.055,.030,"主轴保留全部误差峰；插图若出现仅放大低误差区。N 较少的尾部不能据此判定突然更准。不同候选并非只保留共同成功样本。",
        fontsize=10.2,color="#566470")
    out.mkdir(parents=True,exist_ok=True)
    files=[]
    for extension in ("png","pdf"):
        path=out/f"{stem}.{extension}"
        fig.savefig(path,dpi=175,bbox_inches="tight",facecolor="white")
        files.append(str(path))
    plt.close(fig)
    return {"stem":stem,"metric":metric,"axis":axis,"split":split,"files":files,"panels":panels,
        "validation_selected":validation_selected,"score_reference_100ms_line_is_actual_resume":False,"smoothing":False,
        "main_axis_peak_cropped":False,"aggregation_performed":False}


def _write_ranking(summary,out):
    rows=summary.get("candidate_ranking",[])
    if not rows:
        return None
    fields=list(dict.fromkeys(k for row in rows for k in row))
    path=out/"validation_test_candidate_ranking.csv"
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return str(path)


def _load_summary(summary_path):
    """Restore published compact metric shards when the runtime summary is absent."""
    if summary_path.exists():
        raw=summary_path.read_bytes()
        return json.loads(raw),{str(summary_path):hashlib.sha256(raw).hexdigest()},"full_runtime_summary"
    compact=summary_path.parent/"comparison_summary.json"
    manifest_path=summary_path.parent/"curve_files_manifest.json"
    raw,manifest_raw=compact.read_bytes(),manifest_path.read_bytes()
    summary,manifest=json.loads(raw),json.loads(manifest_raw)
    if manifest.get("schema")!="HEIGHT_POLICY_CURVE_FILES/1":
        raise ValueError("Unsupported height-policy curve manifest")
    hashes={str(compact):hashlib.sha256(raw).hexdigest(),str(manifest_path):hashlib.sha256(manifest_raw).hexdigest()}
    curves=[]
    for spec in manifest["files"]:
        path=manifest_path.parent/spec["path"]
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Metric shard is outside published result directory")
        blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
        if digest!=spec["sha256"]:
            raise ValueError(f"Metric shard hash mismatch: {spec['path']}")
        item=json.loads(blob)
        if (item["split"]!=spec["split"] or item["candidate"]!=spec["candidate"]
            or len(item["cells"])!=spec["cells"]):
            raise ValueError("Metric shard identity/count mismatch")
        curves.extend(item["cells"]);hashes[str(path)]=digest
    summary["curves"]=curves
    return summary,hashes,"compact_comparison_summary_and_verified_curve_shards"


def render(summary_path,out,split="test"):
    if platform.system()!="Linux" or Path(sys.prefix).resolve()!=(ROOT/".venv-tennis").resolve():
        raise RuntimeError("Formal rendering only on Ubuntu in project .venv-tennis")
    summary,source_hashes,source_kind=_load_summary(summary_path)
    font=_style()
    candidates=_selection(summary)
    figures=[]
    for axis,stem,title in [
        ("elapsed20","height_policy_fullstream_elapsed","从首帧开始的全程预测：每策略验证最优模型"),
        ("contact20","height_policy_fullstream_contact","触地对齐的全程预测：每策略验证最优模型"),
    ]:
        for metric,suffix in (("rmse_m",""),("p95_m","_p95")):
            figures.append(_panel_figure(summary,out,split,candidates,axis,stem+suffix,title,metric,validation_selected=True))
    samegp=[(f"gp:{p}",POLICY_NAMES[p],COLORS[i]) for i,p in enumerate(POLICIES[:5])]
    for metric,suffix in (("rmse_m",""),("p95_m","_p95")):
        figures.append(_panel_figure(summary,out,split,samegp,"contact_dense2p5",
            "gp_height_contact_recovery_dense"+suffix,
            "接触与恢复 0–250 ms：相同 GP 碰撞映射，2.5 ms 分桶",metric,(0,.25)))
    known={c["candidate"] for c in summary["curves"] if c["split"]==split}
    comparison=[("gp:continuous","GP · 原持续在线",COLORS[0]),
                ("gp:height_online","GP · 高度在线",COLORS[1]),
                ("gp:hold100","GP · 高度冻结100ms后在线",COLORS[2]),
                ("legacy_ekf6_mlp","旧 EKF6 + MLP · 持续在线",COLORS[4])]
    comparison=[c for c in comparison if c[0] in known]
    if len(comparison)>=2:
        for metric,suffix in (("rmse_m",""),("p95_m","_p95")):
            figures.append(_panel_figure(summary,out,split,comparison,"contact20",
                "gp_height_vs_legacy_mlp"+suffix,
                "GP 策略与旧 MLP 对照：20 ms 共同评分时钟",metric))
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest for path,digest in source_hashes.items()):
        raise RuntimeError("Metric source changed during rendering")
    metadata={"schema":"TENNIS_HEIGHT_POLICY_PLOTS/1","hostname":platform.node(),
        "python_prefix":sys.prefix,"font":font,"source_kind":source_kind,"metric_source_sha256":source_hashes,
        "summary_sha256":source_hashes.get(str(summary_path)),"summary_unchanged":True,"selection_source":"validation_only_selection.per_policy",
        "validation_only_selection":summary["validation_only_selection"],
        "primary_candidates":[c[0] for c in candidates],"figures":figures,
        "ranking_csv":_write_ranking(summary,out),"no_model_training":True,
        "legacy_not_dense":"Legacy MLP is absent from contact_dense2p5 figures and not used to compare dense peaks"}
    (out/"height_policy_plot_metrics.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",default="test")
    args=parser.parse_args()
    report=render(args.summary,args.output,args.split)
    print(json.dumps({"figures":len(report["figures"]),"primary_candidates":report["primary_candidates"],
        "font":report["font"],"summary_sha256":report["summary_sha256"]},ensure_ascii=False))


if __name__=="__main__":main()
