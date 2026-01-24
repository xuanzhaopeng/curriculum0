
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/configs"


echo "Start training curriculum"
echo "Loading configs $CONFIG_PATH"

CUDA_VISIBLE_DEVICES=0 python -m curriculum.start_test \
    --config-path="$CONFIG_PATH" \
    --config-name='curriculum_config' \
    hydra.job.chdir=False

echo "curriculum agent training finished"
