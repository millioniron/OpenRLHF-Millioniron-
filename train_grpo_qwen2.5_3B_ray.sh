export CUDA_VISIBLE_DEVICES=0,1

# 1. 清理环境
# ray stop --force

# 在容器中启动 Ray 的主节点
RAY_memory_monitor_refresh_ms=0 ray start --head --node-ip-address 0.0.0.0 --num-gpus 2 --port 9510



# MODEL_NAME_OR_PATH="examples/test_scripts/checkpoint/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_dr_grpo"

MODEL_NAME_OR_PATH="../Qwen2.5-3B"


# GRPO

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${MODEL_NAME_OR_PATH} \
   --remote_rm_url  examples/scripts/reward_func.py \
   --save_path  examples/test_scripts/checkpoint/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_grpo_k2-0_noformat \
   --micro_train_batch_size 2 \
   --train_batch_size 512 \
   --micro_rollout_batch_size 4  \
   --rollout_batch_size 64  \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --init_kl_coef 0 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 3e-6 \
   --prompt_data ../data/MATH-8k \
   --input_key problem \
   --label_key solution \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 133 \
   --ckpt_path openrlhf/examples/test_scripts/ckpt/Qwen2.5-3B-MATH8k-Train_batch512-rollout_batch64-n_samples_per_prompt=8_lr3e-6_kl=1e-3_replace_onpolicy_grpo_k2-0__noformat \
   --flash_attn \
   --use_wandb 59fa28cc43cac480c8f856677e3370bd423292c5 \
   --colocate_actor_ref  \
   --num_episodes 2 \
   --wandb_project MATH-8k \
   --adam_offload \
#  --flash_attn \
#  --normalize_reward \


# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline