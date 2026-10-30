# 实测机器人与 Vicon 拍心回放

入口：`tools/replay_measured_robot.py`。默认读取 `/home/ubuntu/Downloads/20261020_5组扣杀球_15条轨迹/下位机轨迹` 的五个 CSV，使用已有 X4_950 场景。

## 打开回放

```bash
/home/ubuntu/works/mojuco-mpc/mpc-github/.venv/bin/python \
  /home/ubuntu/works/mojuco-mpc/mpc-github/tools/replay_measured_robot.py
```

默认第 1 组、0.25 倍速、循环播放。空格暂停/继续，R 从头播放，数字 1–5 切换组；鼠标旋转和缩放。也可使用 `--group 4 --speed 0.1`。图例：青色为当前模型拍心，粉色为 Vicon 拍心，黄色为下位机日志计算拍心，白色短线表示当前模型与 Vicon 的位置误差。完整轨迹线仅用于回放展示。

只分析五组并导出所选组的视频：

```bash
/home/ubuntu/works/mojuco-mpc/mpc-github/.venv/bin/python \
  /home/ubuntu/works/mojuco-mpc/mpc-github/tools/replay_measured_robot.py \
  --headless --group 1 \
  --output /tmp/measured_robot_analysis \
  --video /tmp/measured_robot_group1.mp4
```

输出目录必须为空或不存在；视频及同名 PNG 必须不存在。`--video` 使用已安装的 ffmpeg，生成 MP4 和带图例的首帧 PNG。无窗口渲染采用 EGL。首次窗口启动也会自动分析全部五组，结果保存到 `outputs/measured_robot_replay/<时间>/`，控制台打印报告路径。

## 输入与状态契约

- 按表头名称读取制表符/空白分隔的日志，保留 `#` 的列位置；识别这批数据中没有表头名称的末尾一列。异常布局、非有限数值、重复或倒序时间戳直接报错。
- 统一使用 `local_timestamp`（ms），每组第一条下位机记录为显示时间零点；原始时间戳完整保留在比较 CSV。没有按 `dt` 列推进，也不按文件名对时。
- 底盘 XY 乘 0.01，朝向乘 π/180；世界轴保持同向，使用日志中的实际朝向。底盘高度取模型 home 值，roll/pitch 设零。
- 日志五轴位置按度处理；模型关节顺序为 Joint1–Joint5，`q_model = radians(sign * q_log + offset_deg)`。默认方向 `[-1,-1,+1,+1,+1]`，偏移全零。
- CSV 中 `vicon_racket_xyz` 和 `racket_pos_xyz` 按米处理。上述单位由数值和模型 FK 一致性支持；尚未找到下位机 CSV 写入源代码，不能从上位机传输包单位直接推导日志单位。
- 状态回放仅写位姿并调用 `mj_forward`，不调用 `mj_step`、MPC 或电机控制器。`qvel` 为零，画面运动来自连续位姿更新，不输出动力学或拍心速度验证结论。日志没有轮角，轮子/舵角保持模型 home 状态。
- 可以复制输出的 `mapping.json` 并通过 `--mapping` 加载不同方向、顺序、零偏和单位比例；脚本不会自动应用拟合或时间平移，也不修改原模型、数据或原生 MPC 配置。

## 已验证结果（2026-10-28）

结果位置：`outputs/measured_robot_replay/final/`。`report.json` 记录模型和输入文件的 SHA-256、映射、样本数、误差及延迟探查；每组的 `group_N_comparison.csv` 保存原始时间、模型/Vicon/日志 FK 拍心和误差。`group1.mp4` 是第 1 组四分之一速度回放。

| 组 | 模型 vs Vicon，RMSE | 模型 vs 日志 FK，RMSE | 模型 vs Vicon，最大误差 |
|---|---:|---:|---:|
| 1 | 24.90 mm | 0.416 mm | 44.05 mm |
| 2 | 25.57 mm | 0.401 mm | 33.92 mm |
| 3 | 33.65 mm | 0.416 mm | 51.89 mm |
| 4 | 37.27 mm | 0.381 mm | 67.99 mm |
| 5 | 21.54 mm | 0.401 mm | 34.48 mm |

方向探查固定关节顺序与零偏，只用第 1 组日志 FK 对比 32 个正负号组合，最优前四轴为 `[-1,-1,+1,+1]`；J5 两种符号完全等价。其余四组采用相同映射仍有约 0.4 mm 的 FK 一致性。若五轴全部按正方向输入，五组模型/Vicon RMSE 为约 266–517 mm。忽略底盘朝向时，模型/日志 FK RMSE 增至约 2.9–27.8 mm。

这些结果支持当前单位、前四轴方向、底盘朝向及模型 FK 与下位机 FK 的一致性。日志 FK 与模型可能使用同源几何，因此这不是独立的实机几何精度认证。独立 Vicon 对照仍有约 2–4 cm 的 RMSE，不能宣称精确对齐。

### 时间与不可辨识项

时间探查在固定公共内区间内扫描 -40 至 +40 ms，定义为 `model(t + shift)` 对比 `Vicon(t)`。五组最优位移分别为 -1、-18、-6、+3、-14 ms，仅是诊断值，均未应用。组间结果不一致，且部分记录运动很小；几何偏差、反馈滤波和数据年龄差都可能影响该结果，不能作为统一通信延迟标定。

J5 的旋转轴通过拍心，位置观测无法辨识其方向和零位。默认 +1 只是保留当前约定，不能据此认为拍面姿态已验证。Vicon pitch/yaw 的姿态约定尚未核对，本脚本不叠加 Vicon 拍面法向。关节顺序和零偏也只经过当前数据的一致性检查，没有证明唯一解。

本批数据没有单独的 Vicon 采集时间戳，只有本地日志时间；缺少底盘高度/俯仰/横滚和各轮状态。残差仍可能来自传感器不同步、定位参考点、模型几何、标记点偏移或结构形变，本次没有擅自拟合常量位移或修改杆长消除误差。

## 验证

```bash
cd /home/ubuntu/works/mojuco-mpc/mpc-github
.venv/bin/python -m unittest discover -s tests -p 'test_measured_robot_replay.py' -v
```

6 项定向测试通过：混合单位和绝对时钟转换、角度跨界、关节顺序/偏移、错误日志拒绝、已知时间偏移的符号和公共区间、实际模型 FK 与 J5 位置不可辨识性。全部五组完成分析，MP4/PNG 渲染并检查通过；桌面窗口已验证打开、切换第 2/5 组、暂停、重播、继续和正常关闭。
