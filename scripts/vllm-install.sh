source /workspace/.miniconda3/etc/profile.d/conda.sh


cd /workspace

VERL_DIR="/workspace/verl"
VERL_TAG="v0.7.0"

if [ ! -d "$VERL_DIR" ]; then
    git clone --branch $VERL_TAG https://github.com/volcengine/verl.git
else
    cd $VERL_DIR
    git checkout $VERL_TAG
fi

cd /workspace/verl
if ! conda env list | grep -q "^verl"; then
    conda create -n verl python==3.12
fi
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh