#!/usr/bin/env bash
# 每分钟自动拉取 main 最新代码（win 机专用，配合双机协作：mac 侧推送后本机自动跟进）。
# 由 Windows 计划任务 StatusAwareNetwork_AutoPull 经 tools/auto_pull_hidden.vbs 隐藏窗口调用。
#
# 安全策略（最简，绝不丢弃本地修改）：
#   - 不在 main 分支                 → 直接退出（不动当前分支）；
#   - 工作区有未提交修改（已跟踪文件）→ 跳过并记日志；
#   - fetch 后远端无新提交           → 静默退出；
#   - 仅 --ff-only 快进合并；本地与远端分叉（如有未推送本地提交）→ 跳过并记日志。
#
# 日志：results/auto_pull.log（gitignored），只记事件不记空转，避免膨胀。
#
# 计划任务管理（Git Bash）：
#   立即执行：schtasks //Run //TN StatusAwareNetwork_AutoPull
#   暂停：    schtasks //Change //TN StatusAwareNetwork_AutoPull //DISABLE
#   恢复：    schtasks //Change //TN StatusAwareNetwork_AutoPull //ENABLE
#   卸载：    schtasks //Delete //TN StatusAwareNetwork_AutoPull //F

set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG="$REPO/results/auto_pull.log"
mkdir -p "$REPO/results"
cd "$REPO" || exit 1

ts() { date '+%F %T'; }

# 仅在 main 分支上自动同步
if [ "$(git branch --show-current)" != "main" ]; then
  exit 0
fi

# 工作区脏时跳过，避免 pull 覆盖未提交修改
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[$(ts)] skip: 工作区有未提交修改，未拉取" >>"$LOG"
  exit 0
fi

# fetch 失败多为网络/SSH 问题，记日志后本轮放弃，下一分钟自动重试
if ! git fetch origin main >>"$LOG" 2>&1; then
  echo "[$(ts)] fetch 失败（网络/SSH？），跳过本轮" >>"$LOG"
  exit 0
fi

local_sha=$(git rev-parse HEAD) || exit 1
remote_sha=$(git rev-parse origin/main) || exit 1
[ "$local_sha" = "$remote_sha" ] && exit 0  # 无更新，静默

if git merge --ff-only "origin/main" >>"$LOG" 2>&1; then
  echo "[$(ts)] 已快进 ${local_sha:0:7} → ${remote_sha:0:7}" >>"$LOG"
else
  echo "[$(ts)] skip: 本地与 origin/main 分叉，需手动处理" >>"$LOG"
fi
