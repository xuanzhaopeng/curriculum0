import os
import json
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm
from scipy.spatial import ConvexHull

def plot_question_diversity():
    # 1. Gather all question files
    # Search in ./questions/ and ../questions/
    files = glob.glob("questions/results_sc_*.json") + glob.glob("../questions/results_sc_*.json")
    
    if not files:
        print("No question files found in ./questions/ or ../questions/")
        return

    # 2. Extract questions and their scores
    question_to_score = {}
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    q = entry.get("question", "").strip()
                    score = entry.get("self_consistency_score", 0.0)
                    if q:
                        # If a question appears multiple times, we'll keep the latest or highest
                        # For now, let's just store the score
                        question_to_score[q] = float(score)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    questions_list = list(question_to_score.keys())
    scores_list = [question_to_score[q] for q in questions_list]
    
    if not questions_list:
        print("No valid questions found to plot.")
        return

    print(f"Found {len(questions_list)} unique questions. Generating embeddings...")

    # 3. Generate Embeddings
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(questions_list, show_progress_bar=True)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # 4. Dimensionality Reduction (t-SNE)
    print("Performing t-SNE dimensionality reduction...")
    n_samples = len(embeddings)
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    if n_samples > 1:
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=42, 
            init='pca', 
            learning_rate='auto'
        )
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = np.zeros((1, 2))

    # 5. Plotting
    print("Generating plot...")
    plt.figure(figsize=(13, 10))
    
    # Use scores for coloring
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        alpha=0.7, 
        c=scores_list, 
        cmap='viridis',
        edgecolors='w', 
        s=60
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Self-Consistency Score')
    
    plt.title(f"Question Diversity & Performance (t-SNE)\nTotal Unique Questions: {len(questions_list)}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    # Identfy points in the 0.4 - 0.6 range
    mask_mid = [(s >= 0.4 and s <= 0.6) for s in scores_list]
    mid_points = embeddings_2d[mask_mid]
    
    # Draw Area (Convex Hull) for the mid-range points
    if len(mid_points) >= 3:
        try:
            hull = ConvexHull(mid_points)
            # Plot the hull as a shaded area
            for simplex in hull.simplices:
                plt.plot(mid_points[simplex, 0], mid_points[simplex, 1], 'r--', alpha=0.5)
            
            plt.fill(mid_points[hull.vertices, 0], mid_points[hull.vertices, 1], 'r', alpha=0.1, label='Uncertainty Area (0.4-0.6 SC)')
        except Exception as e:
            print(f"Could not draw hull: {e}")
    elif len(mid_points) > 0:
        # Just circle them if there are too few for a hull
        plt.scatter(mid_points[:, 0], mid_points[:, 1], s=200, facecolors='none', edgecolors='r', linewidths=2, label='Uncertainty Points')

    # Annotate random points with their scores
    if len(questions_list) > 0:
        num_to_annotate = min(15, len(questions_list))
        indices = np.random.choice(len(questions_list), num_to_annotate, replace=False)
        for idx in indices:
            score_val = scores_list[idx]
            # If it's in the uncertainty range, make the label more prominent
            is_uncertain = 0.4 <= score_val <= 0.6
            plt.annotate(
                f"SC: {score_val:.2f}", 
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=9,
                fontweight='bold' if is_uncertain else 'normal',
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow' if is_uncertain else 'white', alpha=0.7, ec='red' if is_uncertain else 'gray')
            )
    
    plt.legend(loc='upper right')

    # 6. Save result
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "question_diversity.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diversity plot saved to: {output_file}")

if __name__ == "__main__":
    plot_question_diversity()
