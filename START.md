# 显示模型

```bash
cd /home/ubuntu/works/mojuco-mpc/mpc-github

# 用最新 URDF 和配置重新生成模型
.venv/bin/python sckd_sim/models/build_model.py --variant x4_950

# 打开可视化窗口
.venv/bin/python -m mujoco.viewer --mjcf=sckd_sim/models/generated/x4_950_scene.mjcf
```

## 启动

默认以 0.25 倍速实时显示：

```bash
python3 /home/ubuntu/works/mojuco-mpc/mpc-github/tools/run_x4_native_replay.py --view
```

更慢可在末尾追加 `--view-speed 0.1`。关闭窗口后重新运行同一命令，即可从轨迹起点重置执行。
