CKPT_PATH=/share/home/sxjiang/myproject/ARPO-OCT/ARPO/verl_checkpoints/ARPO_global_16_init_8_beam_2_random_0_arpo_0.2_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth
BASE_MODEL=/share/home/sxjiang/myproject/ARPO-OCT/sft_checkpoints/Qwen2.5-3B-Instruct-arpo_final_sft_edition10-52
TARGET_MODEL=/share/home/sxjiang/myproject/ARPO-OCT/transfer_checkpoints/ARPO_global_16_init_8_beam_2_random_0_arpo_0.2_entropy_oct_downprogressive_em_score_seq_mean_specific_smooth
# 使用空格分隔的数字列表
for i in 78; do

    checkpoint_dir="${CKPT_PATH}/global_step_${i}/actor"

    BASE_MODEL="${BASE_MODEL}"

    target_dir="${TARGET_MODEL}/global_step_${i}/hf"

    echo "Processing step $i..."
    echo "Checkpoint dir: $checkpoint_dir"
    echo "Target dir: $target_dir"

    python3 /share/home/sxjiang/myproject/ARPO-OCT/ARPO/merge_ckpt/convert_checkpoint_from_verl_to_hf.py merge \
        --backend "fsdp" \
        --hf_model_path "$BASE_MODEL" \
        --local_dir "$checkpoint_dir" \
        --target_dir "$target_dir"

done


