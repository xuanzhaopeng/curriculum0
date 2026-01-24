
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

export USE_FLASH_ATTENTION=0
export USE_FUSED_KERNELS=0
export VLLM_ATTENTION_BACKEND=XFORMERS  # 或者 FLASH_ATTN，但如果挂了就换 XFORMERS

# 1. 禁用 P2P，强制走 PCIe 总线（稍微慢一点点，但绝对稳，能解决卡死）
export NCCL_P2P_DISABLE=1

# 2. 禁用 IB，强制走 Ethernet（防止找不到 InfiniBand 卡死）
export NCCL_IB_DISABLE=1

# 3. 打印详细日志（如果还卡，我们能看到卡在哪一步）
export NCCL_DEBUG=INFO

export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Start training curriculum"
echo "Loading configs $CONFIG_PATH"

CUDA_VISIBLE_DEVICES=0,1 python -m curriculum.start_test \
    --config-path="$CONFIG_PATH" \
    --config-name='curriculum_config' \
    hydra.job.chdir=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    trainer.experiment_name=4b_base \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.engine_kwargs.vllm="{max_num_seqs: 1024, max_model_len: 8192}" \
    actor_rollout_ref.rollout.enforce_eager=True

echo "curriculum agent training finished"
