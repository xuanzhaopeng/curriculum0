source /workspace/miniconda3/etc/profile.d/conda.sh


cd /workspace
git clone --branch v0.7.0 https://github.com/volcengine/verl.git
cd /workspace/verl
conda create -n verl python==3.12
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh