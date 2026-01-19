# Agent0 Curriculum Agent


## Pre-requisite in Runpod
```bash
# ssh login WITHOUT TCP
bash scripts/ssh-config.sh

# Then re-login from TCP in VSCode if necessary
```

## Install Dependencies
```bash
huggingface-cli login
bash scripts/verl-install.sh
pip install -r requirements.txt

# (Optional) to check
pip install nvitop
```

## Start Testing
```bash
bash scripts/start_test.sh
```