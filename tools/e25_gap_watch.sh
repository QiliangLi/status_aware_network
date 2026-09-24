#!/bin/zsh
# 等待新格点落地：默认基线 4 格，出现 >4 格即退出
BASE=${1:-4}
for i in $(seq 1 19); do
  sleep 30
  N=$(.venv/bin/python tools/e25_ol_gap_rank.py 2>/dev/null | head -1 | grep -oE '[0-9]+')
  if [ -n "$N" ] && [ "$N" -gt "$BASE" ]; then
    echo "NEW: $N 格点"
    .venv/bin/python tools/e25_ol_gap_rank.py 2>/dev/null | sed -n '2,9p'
    exit 0
  fi
done
echo "TIMEOUT: 仍 $BASE 格点"
