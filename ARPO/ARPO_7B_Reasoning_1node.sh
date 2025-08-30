#!/bin/bash
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# PARENT_DIR="$(dirname "$SCRIPT_DIR")"
# cd "$PARENT_DIR"
# echo "Switched to parent directory: $PARENT_DIR"


# ============================ Environment Setup ============================
# Set basic environment variables
export PYTHONUNBUFFERED=1            
export HYDRA_FULL_ERROR=1           
export VLLM_ATTENTION_BACKEND=XFORMERS 
export VERL_LOGGING_LEVEL=DEBUG
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export RAY_DEBUG_MODE="1"
export WANDB_MODE=offline
# Set Python path
export PYTHONPATH=/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/verl_arpo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="reasoning_tasks" # Modify experiment group
EXPERIMENT_NAME="ARPO_global_16_init_8_beam_2_random_0_arpo_0.2_entropy_debug" # Modify experiment name

# Configuration file path
CONFIG_PATH="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/scripts/config" # Modify the absolute path of the config folder, relative path is not recommended
CONFIG_NAME="ppo_trainer.yaml"

# Distributed training settings
NNODES=1                            
N_GPUS_PER_NODE=2                 

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                 # Prompt field name
TRAIN_BATCH_SIZE=4                # Training batch size
PPO_MINI_BATCH_SIZE=2              # PPO mini-batch size
MAX_PROMPT_LENGTH=1536              # Maximum prompt length
MAX_RESPONSE_LENGTH=4096            # Maximum response length

# Data file paths
TRAIN_FILES="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/rl_datasets/train_10k_boxed_budget.parquet" # Modify training data path
VALID_FILES="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/rl_datasets/valid.parquet" # Modify validation data path

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="/home/sxjiang/myproject/agent/Tool-Star-OCT/transfer_checkpoints/Qwen2.5-3B-Instruct-final_sft_edition10-52" # Modify training model path

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool"       # Synchronous mode with tool support
ROLLOUT_N=8                         # Number of responses generated per sample
INITIAL_ROLLOUTS=4                 # Initial rollout number
BEAM_SIZE=2                        # Beam size
BRANCH_PROBABILITY=0.5             # Branch probability
Entropy_weight=0.2
CALL_LIMIT=3

# ============================ OCT Configuration ==========================
USE_OCT_COFFICIENT=True
OCT_COEF=1
NO_POSITIVE_PENALTY=True
OCT_PENALTY="budget"
PROGRESSIVE_CALLING_TIMES_STAGES=5500
# PROGRESSIVE_MAX

# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/search_cache/search_cache.json" # Modify

# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py" # Modify reward function path
# CUSTOM_REWARD_FUNCTION_NAME="compute_score"
CUSTOM_REWARD_FUNCTION_NAME="compute_score_binary_mix"
# MATH_RULE=binary_f1
MATH_RULE=em_score
QA_RULE="em_score"
MIX_RULES=True
IS_MULTI_TOOL=True
BINARY_F1_THRESHOLD=0.5

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=2                      # Total training epochs
SAVE_FREQ=5                        # Save frequency
TEST_FREQ=5                        # Test frequency

# ============================ Path Configuration ============================
# Save path
SAVE_PATH="/home/sxjiang/myproject/agent/ARPO-OCT/ARPO/verl_checkpoints/${EXPERIMENT_NAME}" # Modify save path
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
# WandB settings
WANDB_API_KEY="" # Modify your wandb key
SEARCH_CLASS_PATH="verl.workers.rollout.tools.search_tool.WikiSearchTool"
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training ============================
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.oct_penalty=${OCT_PENALTY} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_oct_cofficient=${USE_OCT_COFFICIENT} \
    actor_rollout_ref.actor.oct_coef=${OCT_COEF} \
    actor_rollout_ref.actor.no_positive_penalty=${NO_POSITIVE_PENALTY} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path=${SEARCH_CLASS_PATH} \
    actor_rollout_ref.rollout.tools.call_limit=${CALL_LIMIT} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.math_rule=${MATH_RULE} \
    reward_model.qa_rule=${QA_RULE} \
    reward_model.mix_rules=${MIX_RULES} \
    reward_model.is_multi_tool=${IS_MULTI_TOOL} \
    reward_model.binary_f1_threshold=${BINARY_F1_THRESHOLD} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.progressive_calling_times_stages=${PROGRESSIVE_CALLING_TIMES_STAGES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log