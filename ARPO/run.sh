#!/bin/bash
source ~/.bashrc
source ~/miniconda3/bin/activate


for i in {1..10}  
do
cd ~
python kill.py
cd //share/home/sxjiang/myproject/Tool-Star-OCT/
bash ./retriever_launch_hit.sh &
SERVER_PID=$!
echo "ReCall server pid: $SERVER_PID"
# 捕获 Ctrl-C，确保 server 一起被杀掉
trap 'echo "Stopping server..."; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit' INT TERM EXIT

conda activate arpo
cd /home/sxjiang/myproject/agent/ARPO-OCT/ARPO/
bash ./ARPO_7B_Reasoning_1node.sh

done