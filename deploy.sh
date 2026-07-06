#!/bin/bash

LOG_DIR=logs
mkdir -p $LOG_DIR

echo "🚀 启动 viapi.py 和 auapi.py"

# 后台运行 + 日志输出
python viapi.py > $LOG_DIR/viapi.log 2>&1 &
PID1=$!

python auapi.py > $LOG_DIR/auapi.log 2>&1 &
PID2=$!

echo "viapi PID: $PID1"
echo "auapi PID: $PID2"

# 等待两个进程结束
wait $PID1
wait $PID2

echo "✅ 所有任务完成"