
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

# === 支持双卡通信 ===
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start training curriculum"
echo "Loading configs $CONFIG_PATH"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 python -m curriculum.start_test \
    --config-path="$CONFIG_PATH" \
    --config-name='curriculum_config' \
    hydra.job.chdir=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    trainer.experiment_name=4b_base \
    trainer.total_epochs=20 \
    trainer.total_training_steps=20 \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.rollout.engine_kwargs.vllm="{max_num_seqs: 64, max_model_len: 2560}" \
    +actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.exclude_modules=null

echo "curriculum agent training finished"
