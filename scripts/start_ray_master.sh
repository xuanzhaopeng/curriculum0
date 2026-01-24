#!/bin/bash

# Master node environment variables
export GF_SERVER_HTTP_PORT=3000                     # Grafana service default port (customizable)
export PROMETHEUS_PORT=9090                         # Prometheus service default port (customizable)
export RAY_HEAD_PORT=6379                           # Ray master node port (customizable)
export RAY_DASHBOARD_PORT=8265                      # Ray dashboard default port (customizable)
export GRAFANA_PATHS_DATA=/tmp/grafana              # Grafana data storage directory (customizable)
export RAY_GRAFANA_HOST="http://0.0.0.0:${GF_SERVER_HTTP_PORT}"        # Ray-associated Grafana address
export RAY_PROMETHEUS_HOST="http://0.0.0.0:${PROMETHEUS_PORT}"         # Ray-associated Prometheus address
export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1       # Enable actor task logging

# Start Ray on master node
ray start --head --port=${RAY_HEAD_PORT} --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}

# Metrics installation path
INSTALL_DIR="/workspace/tools"

echo "Waiting for Ray to initialize metrics directory..."
sleep 20 # Give Ray a moment to create the session directory

# Overwrite default Grafana config with custom one
if [ -f "./grafana/grafana.ini" ]; then
    echo "Overwriting Grafana config..."
    mkdir -p /tmp/ray/session_latest/metrics/grafana
    echo "====== Original config ======"
    cat /tmp/ray/session_latest/metrics/grafana/grafana.ini
    cp "./grafana/grafana.ini" /tmp/ray/session_latest/metrics/grafana/grafana.ini

    echo "[Important] Grafana config overwritten"
fi

# Start Grafana
if [ -f "$INSTALL_DIR/grafana/bin/grafana-server" ]; then
    echo "Starting Grafana..."
    nohup "$INSTALL_DIR/grafana/bin/grafana-server" \
      --config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
      --homepath "$INSTALL_DIR/grafana" \
      web > grafana.log 2>&1 &
else
    echo "Grafana binary not found at $INSTALL_DIR/grafana/bin/grafana-server"
fi

# Start Prometheus
mkdir -p /workspace/tools/prometheus/data
if [ -f "$INSTALL_DIR/prometheus/prometheus" ]; then
    echo "Starting Prometheus..."
    nohup "$INSTALL_DIR/prometheus/prometheus" \
      --config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml \
      --storage.tsdb.path=/workspace/tools/prometheus/data \
      --storage.tsdb.retention.time=30d \
      --web.enable-lifecycle \
      --web.listen-address=:"${PROMETHEUS_PORT}" \
      > prometheus.log 2>&1 &
else
    echo "Prometheus binary not found at $INSTALL_DIR/prometheus/prometheus"
fi
