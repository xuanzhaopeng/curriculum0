
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"

# === 支持双卡通信 ===
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 2. 强制指定网卡和通信协议
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_FAMILY=AF_INET

# 3. 设置分布式训练必要的地址（如果是单机双卡）
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 4. 开启日志以便万一出错时查看
export NCCL_DEBUG=INFO

export VLLM_ENFORCE_EAGER=1
export RAY_ADDRESS='127.0.0.1'
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_MODELSCOPE=False
# 强制不使用编译后的某些 kernel
export VLLM_ATTENTION_BACKEND=XFORMERS


echo "Start training curriculum"
echo "Loading configs $CONFIG_PATH"

CUDA_VISIBLE_DEVICES=0,1 python -m curriculum.start_test \
    --config-path="$CONFIG_PATH" \
    --config-name='curriculum_config' \
    hydra.job.chdir=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    trainer.experiment_name=4b_base \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.n_gpus_per_node=2 \
    data.train_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.engine_kwargs.vllm="{max_num_seqs: 256, max_model_len: 4096}" \
    +actor_rollout_ref.rollout.layered_summon=True \
    +actor_rollout_ref.rollout.min_p=0.3 \
    actor_rollout_ref.rollout.temperature=1.1 \
    +actor_rollout_ref.model.lora_rank=64 \
    +actor_rollout_ref.model.lora_alpha=32 \
    +actor_rollout_ref.model.target_modules=all-linear \
    +actor_rollout_ref.model.exclude_modules=null

echo "curriculum agent training finished"§
