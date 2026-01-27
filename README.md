# Agent0 Curriculum Agent

## Machine#1 (2xCPU with SandboxFusion)

### Install depency and start service
```bash
cd /workspace/Agent0-curriculum
git pull
pip install -r math_agent/requirements.txt
bash scripts/start_math_agent.sh
python math_agent/test_service.py
python math_agent/test_service_parallel.py
```

## Machine#2 (1xA40)

### Runpod configuration
```bash
cd /workspace/Agent0-curriculum
git pull
bash scripts/ssh-config.sh
```

### Install Dependencies
```bash
bash scripts/verl-install.sh
pip install -r requirements.txt
# (Optional) to check
pip install nvitop

bash scripts/install_metrics.sh
```

### Start Distripution Service
```bash
# replace to the CPU machine's TPC 8000 part mapping
export MATH_AGENT_URL=tcp://213.173.111.109:24947
bash scripts/start_disptacher.sh
```

### Start Ray and Metrics
```bash
bash scripts/start_ray_master.sh
```

### Start Testing
```bash
# clean up previous data
rm -rf questions
rm -rf metrics
rm -rf checkpoints/curriculum_agent

# Start training in background with logging
nohup bash scripts/train.sh > train.log 2>&1 &

# Monitor training progress
tail -f train.log
```