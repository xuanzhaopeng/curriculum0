# Agent0 Curriculum Agent


## Pre-requisite in Runpod
```bash
# ssh login WITHOUT TCP
bash scripts/ssh-config.sh

# Then re-login from TCP in VSCode if necessary
```

## Machine#1 (2xCPU with SandboxFusion)

### Install depency and start service
```bash
pip install -r math_agent/requirements.txt
bash scripts/start_math_agent.sh
```

## Machine#2 (1xA40)

### Install Dependencies
```bash
export MATH_AGENT_URL=https://srxdsrinohfl5f-8000.proxy.runpod.net/solve
huggingface-cli login
bash scripts/verl-install.sh
pip install -r requirements.txt

# (Optional) to check
pip install nvitop
```

### Start Testing
```bash
bash scripts/start_test.sh
```