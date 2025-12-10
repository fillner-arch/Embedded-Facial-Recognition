#!/bin/bash

# Name of your Python script
SCRIPT="main.py"

# Start your program in background
python3 "$SCRIPT" &
pid=$!

# Output file names
LOG_FILE="benchmark/resource_${pid}.txt"
PLOT_FILE="benchmark/usage_${pid}.png"

# Record CPU/memory usage
echo "Recording CPU and memory usage for PID $pid..."
psrecord $pid --interval 1 --log "$LOG_FILE" --plot "$PLOT_FILE"

# Wait until process ends
wait $pid
echo "Benchmark finished. Logs saved to $LOG_FILE and plot to $PLOT_FILE."
