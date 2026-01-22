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

    # 2. Extract scores for unique questions
    question_to_score = {}
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    q = entry.get("question", "").strip()
                    score = entry.get("self_consistency_score", 0.0)
                    if q:
                        # Keep the latest score for each unique question
                        question_to_score[q] = float(score)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    scores = list(question_to_score.values())
    if not scores:
        print("No valid scores found to plot.")
        return

    print(f"Found {len(scores)} unique questions. Calculating distribution...")

    # 3. Define bins [0, 0.1, 0.2, ..., 1.0]
    # We round each score to the nearest 0.1 to fit into these buckets
    bins = np.arange(0, 1.1, 0.1)
    counts = np.zeros_like(bins)
    
    for s in scores:
        # Find the index of the nearest bin value
        idx = np.abs(bins - s).argmin()
        counts[idx] += 1

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    # Define colors: red for uncertain bins (0.4, 0.5, 0.6), skyblue for others
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

    plt.title(f"Self-Consistency Score Distribution\nTotal Unique Questions: {len(scores)}")
    plt.xlabel("Self-Consistency Score (Rounded to nearest 0.1)")
    plt.ylabel("Number of Questions")
    plt.xticks(bins)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 5. Save result
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "question_sc_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plot saved to: {output_file}")

if __name__ == "__main__":
    plot_sc_distribution()
