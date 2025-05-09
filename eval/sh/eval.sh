set -ex
export CUDA_VISIBLE_DEVICES=1

# PROMPT_TYPE="qwen25-math-cot"
PROMPT_TYPE="qwen-boxed"

# 定义多个 BASE_MODEL_PATH
BASE_MODEL_PATHS=(
    "/Data/gz/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch256-n_samples_per_prompt=8_lr2e-6_kl=0_offpolicy_dr_grpo_noformat_wstd_overlong/global_step"
    "/Data/gz/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch256-n_samples_per_prompt=8_lr2e-6_kl=0_offpolicy_dr_grpo_noformat_wstd_overlong_p1/global_step"
)


OUTPUT_DIR="./output"
SPLIT="test"
NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME="math500,aime24,gaokao2023en,amc23,minerva_math,olympiadbench"
# DATA_NAME="aime24"

# 外层循环：遍历 global_step 从 10 到 200，步长为 10
for STEP in $(seq 5 5 40); do
    echo "Starting tests for global_step=${STEP}"
    
    # 内层循环：遍历每个 BASE_MODEL_PATH
    for BASE_MODEL_PATH in "${BASE_MODEL_PATHS[@]}"; do
        MODEL_NAME_OR_PATH="${BASE_MODEL_PATH}${STEP}_hf"
        
        TOKENIZERS_PARALLELISM=false \
        python3 -u math_eval.py \
            --model_name_or_path ${MODEL_NAME_OR_PATH} \
            --data_name ${DATA_NAME} \
            --output_dir ${OUTPUT_DIR} \
            --split ${SPLIT} \
            --prompt_type ${PROMPT_TYPE} \
            --num_test_sample ${NUM_TEST_SAMPLE} \
            --max_tokens_per_call 5096 \
            --seed 0 \
            --temperature 0 \
            --n_sampling 1 \
            --top_p 1 \
            --start 0 \
            --end -1 \
            --use_vllm \
            --save_outputs
        
        echo "Finished testing for BASE_MODEL_PATH=${BASE_MODEL_PATH} at global_step=${STEP}"
    done
    
    echo "Completed all BASE_MODEL_PATH tests for global_step=${STEP}"
done

echo "All tests completed."



# set -ex
# export CUDA_VISIBLE_DEVICES=1

# # PROMPT_TYPE="qwen25-math-cot"
# PROMPT_TYPE="qwen-boxed"


# # MODEL_NAME_OR_PATH="../examples/test_scripts/checkpoint/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_dr_grpo"

# MODEL_NAME_OR_PATH="../../ORZ-7B"

# # MODEL_NAME_OR_PATH="../openrlhf/examples/test_scripts/ckpt/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_dr_grpo_k2-0__noformat/_actor/output_dir "

# # MODEL_NAME_OR_PATH="/Data/gz/Qwen2.5-3B-MATH8k-Train_batch256-rollout_batch64-n_samples_per_prompt=8_lr5e-7_kl=0_onpolicy_dr_grpo_k2_noformat_wstd_deter/global_step10_hf"

# OUTPUT_DIR="./output"

# SPLIT="test"
# NUM_TEST_SAMPLE=-1

# # English open datasets
# # DATA_NAME="gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
# DATA_NAME="math500,aime24,gaokao2023en,amc23,minerva_math,olympiadbench"
# # DATA_NAME="minerva_math,olympiadbench"

# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --max_tokens_per_call 5096 \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs 

# # English multiple-choice datasets
# DATA_NAME="aqua,sat_math,mmlu_stem"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --num_shots 5

# # Chinese gaokao collections
# DATA_NAME="gaokao2024_I,gaokao2024_II,gaokao2024_mix,gaokao_math_cloze,gaokao_math_qa"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot

# # Chinese other datasets
# DATA_NAME="cmath,cn_middle_school"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs \
#     --overwrite \
#     --adapt_few_shot


# # English competition datasets
# DATA_NAME="aime24"
# TOKENIZERS_PARALLELISM=false \
# python3 -u math_eval.py \
#     --model_name_or_path ${MODEL_NAME_OR_PATH} \
#     --data_name ${DATA_NAME} \
#     --output_dir ${OUTPUT_DIR} \
#     --split ${SPLIT} \
#     --prompt_type ${PROMPT_TYPE} \
#     --num_test_sample ${NUM_TEST_SAMPLE} \
#     --seed 0 \
#     --temperature 0 \
#     --n_sampling 1 \
#     --top_p 1 \
#     --start 0 \
#     --end -1 \
#     --use_vllm \
#     --save_outputs 