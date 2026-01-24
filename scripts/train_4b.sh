
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"


echo "Start training curriculum"
echo "Loading configs $CONFIG_PATH"

CUDA_VISIBLE_DEVICES=0 python -m curriculum.start_test \
    --config-path="$CONFIG_PATH" \
    --config-name='curriculum_config' \
    hydra.job.chdir=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Base \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.ref.engine_kwargs.vllm.max_model_len=4096 \
    actor_rollout_ref.ref.engine_kwargs.vllm.max_num_seqs=512
    

echo "curriculum agent training finished"
