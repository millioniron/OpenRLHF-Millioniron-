bash sh/eval_single_node.sh \
    --run_name ../examples/test_scripts/checkpoint/Qwen2.5-3B-Train_batch256-rollout_batch32-n_samples_per_prompt=8_lr5e-6  \
    --init_model_path ../examples/test_scripts/checkpoint/Qwen2.5-3B-Train_batch256-rollout_batch32-n_samples_per_prompt=8_lr5e-6 \
    --template qwen25-math-cot  \
    --tp_size 2