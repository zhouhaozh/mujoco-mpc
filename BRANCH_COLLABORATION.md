# 历史与分支说明

当前仓库保留原团队提交图，没有改写旧提交。旧 GitLab 的地址、导入说明、作者信息及历史实验文件仍存在于过去的提交中，当前工作树不再依赖原远端或固定机器目录。

## 保留分支

| 分支 | 内容 |
|---|---|
| `main` | 当前可独立克隆、安装和运行的整理版本 |
| `team/main` | 原团队主线，`3121caaa49b604afc192cf8f9dcff889f681d73c` |
| `codex/tennis-predictor-pfly-nmpc6` | 预测与控制集成分支，`7a95839fc9fb752e402cd66d4cb241f19695f972` |
| `codex/pipeline-modularization` | 同步运行时模块化分支，`9d38dd0fd6d8958b91c020cf3e93dd0819be55db` |
| `codex/x4-inertia-integration-20261026` | X4 惯量集成分支，`adb0c4974d3340a2ca9d4e12533ff500cf4295ce` |
| `archive/source-snapshot-20261020` | 首次源码快照，`a0d34af2d23bc0a099a44154f5dd564e3f39d8c6` |

整理前全部 82 个历史提交的 SHA 和祖先关系保持不变。当前整理仅更新 `main` 的最新快照，移除了环境、构建产物、视频和大体积实验输出；这些内容在已有历史提交中仍可追溯。

## 基于历史分支开发

向 GitHub 上传全部保留分支后，新克隆可以查看远端分支，或从原团队基线建立新分支：

```bash
git branch -a
git switch -c feature/my-work origin/team/main
```

当前本地整理目录已经有六个本地分支，可直接以 `team/main` 为基线；如果 GitHub 只包含 `main`，则需要另外上传历史分支后才能在新克隆中使用它们。仓库 URL 与访问认证应只配置在本地 Git remote 中，不要写入源码、脚本或文档。

## 验证边界

保留提交历史只证明源码来源与祖先关系完整，不代表历史性能报告已在新机器复现。新环境应按根目录 `README.md` 重新安装、构建并运行测试。
