#!/bin/bash
# 激活llama factory环境
source /mmu_nlp_ssd/makai05/miniconda3/bin/activate
conda activate verl

# 进入llama factory所在目录
# cd /mmu_nlp_ssd/dujiazhen/xiaokuai/deepresearch/tool_inference_quick_hand_v2/blc/arpo/LLaMA-Factory

# 基础配置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 可见GPU列表
export PYTHONPATH=$(pwd):$PYTHONPATH

# 关闭wandb
export WANDB_DISABLED=true

#============= 训练参数配置 =============#
# 分布式配置
NNODES=1                 # 节点总数
NODE_RANK=0              # 当前节点rank
PROC_PER_NODE=8          # 每个节点进程数
MASTER_ADDR="127.0.0.1"  # 主节点地址
MASTER_PORT=29500        # 主节点端口

# 输出配置
OUTPUT_DIR="checkpoints/qwen"
# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 训练脚本
TRAIN_SCRIPT="../src/llamafactory/launcher.py"

# 训练参数配置文件
TRAIN_ARGS=""yaml/qwen.yaml""

# 训练执行命令
torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} \
         ${TRAIN_ARGS} 2>&1 | tee ${OUTPUT_DIR}/training.log
         

# 启用日志重定向
# exec >> ${OUTPUT_DIR}/training.log 2>&1