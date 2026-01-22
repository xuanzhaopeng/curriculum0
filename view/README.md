# Metrics Visualization

This folder contains utilities to visualize training metrics collected in JSON format.

## Usage

1. Ensure training has started and `metrics_step_*.json` files are being generated in the project root.
2. Run the plotting script from this directory:
   ```bash
   python view/plot_metrics.py
   ```
3. View the generated plots in the `plots/` subdirectory.

## Requirements
- `matplotlib`
- `numpy`
