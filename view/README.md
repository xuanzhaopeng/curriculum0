# Metrics Visualization

This folder contains utilities to visualize training metrics collected in JSON format.

## Usage

1. Ensure training has started and `metrics_step_*.json` files are being generated in the project root.
2. Run the plotting script from this directory:
   ```bash
   # Download all remote data required for visualization
   rm -rf plots
   bash scripts/download_remote_metrics.sh
   bash scripts/download_remote_questions.sh
   
   # Generate plots
   python view/plot_metrics.py
   ```
3. Run the question diversity plotting script to see the semantic distribution of generated questions:
   ```bash
   python view/plot_question_diversity.py
   ```
4. View the generated plots in the `plots/` subdirectory:
   - `metric_*.png`: Individual trend plots for training metrics.
   - `question_diversity.png`: 2D t-SNE visualization of question embeddings.

## Requirements
- `matplotlib`
- `numpy`
- `sentence-transformers`
