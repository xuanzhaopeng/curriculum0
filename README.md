# Agent0 Curriculum Agent



## Pre-requisite in Runpod
```bash
conda create -n agent0-curriculum python==3.12
conda activate agent0-curriculum

# Instal miniconda
bash scripts/conda-install.sh
# install cuda 12.8
bash scripts/cuda-install.sh
# install cuDNN 9.10.2
bash scripts/cudnn-install.sh
# install vLLM required by VERL
bash scripts/vllm-install.sh
```

## Install Dependencies
```bash
conda activate agent0-curriculum
bash scripts/vllm-install.sh
pip install -r requirements.txt
```