# set -x

export CUDA_VISIBLE_DEVICES=0,1,3


# 1. 清理环境
# ray stop --force

# 在容器中启动 Ray 的主节点
ray start --head  --num-gpus 3 --port 7000



# MODEL_NAME_OR_PATH="examples/test_scripts/checkpoint/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_dr_grpo"

MODEL_NAME_OR_PATH="../Qwen2.5-3B"


# GRPO

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${MODEL_NAME_OR_PATH} \
   --remote_rm_url  examples/scripts/reward_func.py \
   --save_path  examples/test_scripts/checkpoint/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch256-n_samples_per_prompt=8_lr2e-6_kl=0_offpolicy_dr_grpo_noformat_wstd_overlong \
   --micro_train_batch_size 2 \
   --train_batch_size 512 \
   --micro_rollout_batch_size 4  \
   --rollout_batch_size 256  \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 5096 \
   --max_samples 100000 \
   --generate_max_len 5096 \
   --init_kl_coef 0 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator dr_grpo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-6 \
   --prompt_data train_data/MATH-8k \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 5 \
   --ckpt_path /Data/gz/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch256-n_samples_per_prompt=8_lr2e-6_kl=0_offpolicy_dr_grpo_noformat_wstd_overlong \
   --flash_attn \
   --use_wandb {} \
   --colocate_actor_ref  \
   --num_episodes 10 \
   --wandb_project MATH-8k \
   --adam_offload \
   --save_hf_ckpt \
   --disable_ds_ckpt \

   # --full_deterministic \
#  --flash_attn \
#  --normalize_reward \


# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline