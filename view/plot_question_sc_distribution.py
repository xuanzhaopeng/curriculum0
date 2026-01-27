import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_sc_distribution():
    # 1. Gather all question files
    # Search in ./questions/ and ../questions/
    files = glob.glob("questions/results_sc_*.json") + glob.glob("../questions/results_sc_*.json")
    
    if not files:
        print("No question files found in ./questions/ or ../questions/")
        return

    # 5. Output directory
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in files:
        base_name = os.path.basename(f).replace(".json", "")
        print(f"Processing {f}...")
        
        # 2. Extract scores for this specific file
        scores = []
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    score = entry.get("self_consistency_score", 0.0)
                    scores.append(float(score))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

        if not scores:
            print(f"No valid scores found in {f}, skipping.")
            continue

        # 3. Define bins [0, 0.1, 0.2, ..., 1.0]
        bins = np.arange(0, 1.1, 0.1)
        counts = np.zeros_like(bins)
        
        for s in scores:
            idx = np.abs(bins - s).argmin()
            counts[idx] += 1

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        
        bar_colors = []
        for b in bins:
            if 0.39 <= b <= 0.61: # Using range to catch 0.4, 0.5, 0.6 accurately
                bar_colors.append('salmon')
            else:
                bar_colors.append('skyblue')
                
        bars = plt.bar(bins, counts, width=0.08, color=bar_colors, edgecolor='navy', alpha=0.7)
        
        # Add text labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.title(f"Self-Consistency Score Distribution\nFile: {os.path.basename(f)}\nTotal Questions: {len(scores)}")
        plt.xlabel("Self-Consistency Score (Rounded to nearest 0.1)")
        plt.ylabel("Number of Questions")
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        output_file = os.path.join(output_dir, f"{base_name}_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Distribution plot saved to: {output_file}")

if __name__ == "__main__":
    plot_sc_distribution()
