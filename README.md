# Agent0 Curriculum Agent

## Machine#1 (2xCPU with SandboxFusion)

### Install depency and start service
```bash
pip install -r math_agent/requirements.txt
bash scripts/start_math_agent.sh
```

## Machine#2 (1xA40)

## Pre-requisite in Runpod
```bash
# ssh login WITHOUT TCP
bash scripts/ssh-config.sh

# Then re-login from TCP in VSCode if necessary
```

### Install Dependencies
```bash
huggingface-cli login
bash scripts/verl-install.sh
pip install -r requirements.txt

# (Optional) to check
pip install nvitop
```

### Start Distripution Service
```bash
export MATH_AGENT_URL=https://srxdsrinohfl5f-8000.proxy.runpod.net/solve
bash scripts/start_disptacher.sh
```

### Start Testing
```bash
bash scripts/start_test.sh
```