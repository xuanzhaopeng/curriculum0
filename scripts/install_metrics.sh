#!/bin/bash

# Script to install Prometheus and Grafana into /workspace/tools on Ubuntu
INSTALL_DIR="/workspace/tools"
PROMETHEUS_VERSION="2.45.0"
GRAFANA_VERSION="10.0.3"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR" || exit

echo "Target installation directory: $INSTALL_DIR"

# --- Install Prometheus ---
if [ -f "$INSTALL_DIR/prometheus/prometheus" ]; then
    echo "Prometheus is already installed in $INSTALL_DIR/prometheus, skipping."
else
    echo "Installing Prometheus v$PROMETHEUS_VERSION..."
    PROM_URL="https://github.com/prometheus/prometheus/releases/download/v$PROMETHEUS_VERSION/prometheus-$PROMETHEUS_VERSION.linux-amd64.tar.gz"
    
    wget "$PROM_URL" -O prometheus.tar.gz
    tar -xzf prometheus.tar.gz
    mv "prometheus-$PROMETHEUS_VERSION.linux-amd64" prometheus
    rm prometheus.tar.gz
    echo "Prometheus installed successfully."
fi

# --- Install Grafana ---
if [ -f "$INSTALL_DIR/grafana/bin/grafana-server" ]; then
    echo "Grafana is already installed in $INSTALL_DIR/grafana, skipping."
else
    echo "Installing Grafana v$GRAFANA_VERSION..."
    GRAFANA_URL="https://dl.grafana.com/oss/release/grafana-$GRAFANA_VERSION.linux-amd64.tar.gz"
    
    wget "$GRAFANA_URL" -O grafana.tar.gz
    tar -xzf grafana.tar.gz
    mv "grafana-$GRAFANA_VERSION" grafana
    rm grafana.tar.gz
    echo "Grafana installed successfully."
fi

echo "------------------------------------------------"
echo "Installation complete."
echo "Prometheus: $INSTALL_DIR/prometheus/prometheus"
echo "Grafana:    $INSTALL_DIR/grafana/bin/grafana-server"
echo "------------------------------------------------"
