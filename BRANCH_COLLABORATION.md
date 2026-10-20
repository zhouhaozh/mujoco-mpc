# GitLab分支与团队协作

日期：2026-10-20；代码唯一来源：Ubuntu /home/ubuntu/Zhaohui/sim。

仓库：http://172.20.0.32/LZH/mpc （私有）。

## 分支职责

| GitLab分支 | 内容 |
|---|---|
| main | 当前模块化代码，加一次关联初次GitLab快照的合并 |
| team/main | 原团队main，3121caaa49b604afc192cf8f9dcff889f681d73c |
| codex/tennis-predictor-pfly-nmpc6 | 原预测分支，7a95839fc9fb752e402cd66d4cb241f19695f972 |
| codex/pipeline-modularization | 在原预测分支上提交当前Ubuntu改动 |
| archive/source-snapshot-20261020 | 初次独立上传快照，a0d34af2d23bc0a099a44154f5dd564e3f39d8c6 |

原73个提交的SHA保持不变，无tags。原预测分支领先原team/main 9个提交；模块化提交直接以预测分支为父。
默认main的第一次关联合并以模块化提交和初次快照为双父，最终文件树与模块化提交相同。
这使现有GitLab main可以正常快进，且原团队祖先关系保留；没有force push或历史重写。
该次导入合并只用于连接既有两条历史，后续团队协作按常规merge处理。

## 新成员克隆与切换

~~~bash
git clone http://172.20.0.32/LZH/mpc.git
cd mpc
git switch --track origin/codex/pipeline-modularization
~~~

新成员如需从原团队基线建立个人分支：

~~~bash
git switch -c feature/my-work origin/team/main
~~~

## 在已有团队仓库合并当前版本

保留自己的origin配置，新增远端名称lzh；如果同名远端已经存在，先检查URL，不重复添加。

~~~bash
git remote add lzh http://172.20.0.32/LZH/mpc.git
git fetch lzh
git switch <自己的工作分支>
git merge --no-ff lzh/codex/pipeline-modularization
~~~

请先保存自己的未提交改动。如果自己的分支仍基于原团队历史，Git能够识别共同祖先；
双方修改同一内容时仍可能有正常冲突，需要结合业务含义解决。后续合并不需要allow-unrelated-histories。

Ubuntu现有工作区后续推送：

~~~bash
git -C /home/ubuntu/Zhaohui/sim push gitlab codex/pipeline-modularization
~~~

认证只在终端交互输入。不要把密码、令牌或私钥写入URL、脚本、Git配置或文档。

## 数据与验证范围

本次同步代码与原Git历史；模型、环境、数据、视频和结果仍由Ubuntu保存。
克隆后须配置外部运行资源；Git历史完整和合并成功不代表已完成构建、300Hz或实机验收。
