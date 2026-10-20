# pfly / nmpc6 控制修正与17轴曲线（2026-10-15）

本轮只评估pfly和nmpc6，wbmpc不参与。完整说明在Ubuntu配套预测项目：

```text
/home/ubuntu/Zhaohui/mpc_sim/docs/tennis_pfly_nmpc6_control_optimization.md
/home/ubuntu/Zhaohui/sim/sckd_sim/artifacts/control_ablation_20261015/paired_v2/report/index.html
```

源码默认对pfly/nmpc6启用`feasible`：终端锚/途中走廊分离，上游加速/转向/臂行程筛选，求解成功＋约束残差＋严格终点验收。原基线通过`M2_CONTROL_VARIANT=baseline`显式恢复。

可选累计候选：`clock`、`feedforward`、`gains`；nmpc6另有`servo_mpc`和`base_feedback`研究候选。增益参数来自当前仿真五轴辨识，放在`models/config/servo_identified_20261015.json`，不是实机标定；力矩包络保持原值。

同35球固定因果预测快照完成385次主配对；增益候选的有限拍面几何拦截nmpc6/pfly为19/35、20/35，原基线13/35、15/35。nmpc6仍有位置尾部和底盘跟踪问题。底盘位置反馈补充70次同二进制配对，拦截19→16，未设为默认。主组与补充组分别保存，不混称同一二进制。两种方法的末端、五臂轴、四舵、八轮曲线及补充合计700页。

入口：`tools/control_ablation_cache.py`（因果预测缓存）、`run_control_ablation.py`（配对运行）、`analyze_control_ablation.py`（误差/几何/曲线）、`validate_control_ablation.py`（独立审计）、`assemble_control_ablation_report.py`（组合浏览页）。`build_servo_nmpc.py`从版本化辨识参数构建可选执行器预测库，不覆盖原nlp.so。

Ubuntu7组CTest通过；455次分组审计、同球输入/缓存哈希、锁止快照及17轴限幅逐条通过；高拦截候选70条0.5ms/1ms几何分类一致。此处仍是实测轨迹外源回放的几何估计，不是实机命中或出球验收；缓存评估也不是实时频率验收。
