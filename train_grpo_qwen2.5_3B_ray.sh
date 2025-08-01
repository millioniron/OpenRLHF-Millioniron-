set -x

# 1. 清理环境
ray stop --force

export CUDA_VISIBLE_DEVICES=7,5,6,4
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# 在容器中启动 Ray 的主节点
ray start --head  --num-gpus 4 --port 7200




MODEL_NAME_OR_PATH="Qwen2.5-Math-1.5B-16k"




# GRPO

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --actor_num_nodes 2 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${MODEL_NAME_OR_PATH} \
   --remote_rm_url  examples/scripts/reward_func.py \
   --save_path  /Data/Qwen2.5-Math-1.5B-16k-LUFFY-distill-all-mixpolicy \
   --micro_train_batch_size 1 \
   --train_batch_size 512 \
   --micro_rollout_batch_size 2  \
   --rollout_batch_size 64 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 8192  \
   --init_kl_coef 0 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k2 \
   --advantage_estimator dr_grpo \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-6 \
   --prompt_data train_data/LUFFY-distill \
   --input_key problem \
   --label_key answer \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 20  \
   --ckpt_path /Data/Qwen2.5-Math-1.5B-16k-LUFFY-distill-all-mixpolicy \
   --flash_attn \
   --use_wandb 59fa28cc43cac480c8f856677e3370bd423292c5 \
   --colocate_actor_ref  \
   --num_episodes 5 \
   --lr_warmup_ratio 0.0001 \
   --wandb_project LUFFY-distill \
   --adam_offload \
   --save_hf_ckpt \
   --disable_ds_ckpt \
   --mixpolicy \

   # --full_deterministic \
#  --flash_attn \
#  --normalize_reward \


# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline