
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

# === 2. 核心修复 (这就是刚才测试通过的原因) ===
# 强制让 VeRL 使用和刚才测试脚本一样的通信配置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO  # 保持日志开启，万一有问题能看到

# === 3. 强制兼容模式 (防止 FlashAttn 导致的 Kernel 死锁) ===
# 虽然机器没坏，但为了稳过，我们先禁用 FlashAttn，跑通再说
export USE_FLASH_ATTENTION=0
export VLLM_ATTENTION_BACKEND=XFORMERS

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
    actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.rollout.engine_kwargs.vllm="{max_num_seqs: 1024, max_model_len: 8192}" \
    actor_rollout_ref.rollout.enforce_eager=True

echo "curriculum agent training finished"
