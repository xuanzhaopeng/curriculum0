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
conda activate agent0-curriculum
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# The pynvml package is deprecated. 
pip uninstall pynvml -y
pip install nvidia-ml-py