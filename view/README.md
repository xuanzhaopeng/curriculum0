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
   ```
3. Run the visualization scripts to see metrics and diversity:
   ```bash
   python view/plot_metrics.py
   python view/plot_question_diversity.py
   python view/plot_question_sc_distribution.py
   python view/extract_perfect_questions.py
   ```
4. View the generated results in the `plots/` subdirectory:
   - `metric_*.png`: Individual trend plots for training metrics.
   - `question_diversity.png`: 2D t-SNE visualization of question embeddings + uncertainty zone.
   - `question_sc_distribution.png`: Bar chart of self-consistency scores.
   - `perfect_questions.json`: Questions with 1.0 SC score.

## Requirements
- `matplotlib`
- `numpy`
- `sentence-transformers`
