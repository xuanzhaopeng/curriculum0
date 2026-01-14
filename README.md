# Agent0 Curriculum Agent


## Start
```bash
conda create -n agent0-curriculum python==3.12
conda activate agent0-curriculum
```

## Runpod prepare
```bash
# Instal miniconda
bash scripts/conda-install.sh
# install cuda 12.8
bash scripts/cuda-install.sh
# install cuDNN 9.10.2
bash scripts/cudnn-install.sh
# install vLLM
bash scripts/vllm-install.sh
```