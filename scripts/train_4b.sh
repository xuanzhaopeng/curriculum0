
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"


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
    actor_rollout_ref.rollout.engine_kwargs.vllm="{max_num_seqs: 1024, max_model_len: 8192}"

echo "curriculum agent training finished"
