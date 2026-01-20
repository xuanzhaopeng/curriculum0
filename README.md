# Agent0 Curriculum Agent

## Machine#1 (2xCPU with SandboxFusion)

### Runpod configuration
```bash
cp -r /workspace/.ssh /root/.ssh
chmod 600 /root/.ssh/id_ed25519
chmod 700 /root/.ssh
```

### Install depency and start service
```bash
cd /workspace/Agent0-curriculum
git pull
pip install -r math_agent/requirements.txt
bash scripts/start_math_agent.sh
```

## Machine#2 (1xA40)

### Runpod configuration
```bash
cp -r /workspace/.ssh /root/.ssh
chmod 600 /root/.ssh/id_ed25519
chmod 700 /root/.ssh
cd /workspace/Agent0-curriculum
git pull
bash scripts/ssh-config.sh
```

### Install Dependencies
```bash
bash scripts/verl-install.sh
pip install -r requirements.txt
bash scripts/install_metrics.sh

# (Optional) to check
pip install nvitop
```

### Start Distripution Service
```bash
export MATH_AGENT_URL=https://srxdsrinohfl5f-8000.proxy.runpod.net/solve
bash scripts/start_disptacher.sh
```

### Start Ray and Metrics
```bash
bash scripts/start_ray_master.sh
```

### Start Testing
```bash
bash scripts/train.sh
```

## RunPod Automation

If you are managing your pods from a local machine, you can use the automation scripts to resume and prepare both machines in the correct order.

1.  **Set Environment Variables**:
    ```bash
    export RUNPOD_API_KEY="your_runpod_api_key"
    export GEMINI_API_KEY="your_gemini_api_key"
    ```
2.  **Run the Setup Script**:
    ```bash
    bash scripts/runpod_setup.sh
    ```
    This script will:
    *   Resume both the CPU (`srxdsrinohfl5f`) and GPU (`2oq01s2u35mms7`) pods.
    *   Wait for them to be ready.
    *   SSH into each to perform the configuration and start the required services.