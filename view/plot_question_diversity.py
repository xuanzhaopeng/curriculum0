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
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Gather all question files
    # Search in ./questions/ and ../questions/
    files = glob.glob("questions/results_sc_*.json") + glob.glob("../questions/results_sc_*.json")
    
    if not files:
        print("No question files found in ./questions/ or ../questions/")
        return

    # 2. Extract questions and their metadata
    question_metadata = {}
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    q = entry.get("question", "").strip()
                    if q:
                        # Store metadata. If duplicate, the last one found wins.
                        question_metadata[q] = {
                            "sc_score": float(entry.get("self_consistency_score", 0.0)),
                            "majority_answer": entry.get("majority_answer", None),
                            "all_answers": entry.get("all_answers", [])
                        }
        except Exception as e:
            print(f"Error reading {f}: {e}")

    questions_list = list(question_metadata.keys())
    scores_list = [question_metadata[q]["sc_score"] for q in questions_list]
    
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
    
    # Create custom colormap: Green (0.0) -> Red (0.5) -> Green (1.0)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#2ecc71", "#e74c3c", "#2ecc71"] # Green, Red, Green
    n_bins = 100
    uncertain_cmap = LinearSegmentedColormap.from_list("uncertainty_cmap", colors, N=n_bins)

    # Use scores for coloring
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        alpha=0.7, 
        c=scores_list, 
        cmap=uncertain_cmap,
        edgecolors='w', 
        s=60,
        vmin=0, vmax=1
    )
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Self-Consistency Score (Red = High Uncertainty)')
    
    plt.title(f"Question Diversity & Performance (t-SNE)\nTotal Unique Questions: {len(questions_list)}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    # Identfy points in the 0.4 - 0.6 range
    mask_mid = np.array([(s >= 0.4 and s <= 0.6) for s in scores_list])
    mid_indices = np.where(mask_mid)[0]
    
    mid_points = embeddings_2d[mask_mid]
    
    # 6. Find distinct Uncertainty Islands
    if len(mid_points) >= 3:
        from sklearn.cluster import DBSCAN
        from matplotlib.path import Path
        
        # Heuristic for clustering distance: small even tighter to find more "clean" islands
        spread = np.ptp(embeddings_2d, axis=0)
        eps = np.mean(spread) * 0.04 # Much smaller to find tighter groups
        
        clustering = DBSCAN(eps=eps, min_samples=3).fit(mid_points)
        labels = clustering.labels_
        
        print(f"🔍 Found {len(mid_points)} uncertain points. DBSCAN labels identified: {set(labels)}")
        
        areas_drawn = 0
        unique_labels = sorted([l for l in set(labels) if l != -1])
        
        for cluster_id in unique_labels:
            cluster_indices_in_mid = np.where(labels == cluster_id)[0]
            cluster_points = mid_points[cluster_indices_in_mid]
            cluster_global_indices = mid_indices[cluster_indices_in_mid]
            
            # JSON Export: Always record every area found
            area_data = []
            for idx in cluster_global_indices:
                q_text = questions_list[idx]
                meta = question_metadata[q_text]
                area_data.append({
                    "question": q_text,
                    "self_consistency_score": meta["sc_score"],
                    "majority_answer": meta["majority_answer"],
                    "all_answers": meta["all_answers"]
                })
            
            area_num = cluster_id + 1
            area_file = os.path.join(output_dir, f"area_{area_num}.json")
            with open(area_file, 'w', encoding='utf-8') as jf:
                json.dump(area_data, jf, indent=2, ensure_ascii=False)
            print(f"📦 Exported Cluster {cluster_id} to: {area_file} ({len(area_data)} questions)")

            # Visualization: Only draw if the area is "Clean" (no other points inside)
            if len(cluster_points) >= 3:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_vertices = cluster_points[hull.vertices]
                    hull_path = Path(hull_vertices)
                    
                    certain_points_mask = ~mask_mid
                    certain_points = embeddings_2d[certain_points_mask]
                    
                    is_clean = True
                    if len(certain_points) > 0:
                        points_inside = hull_path.contains_points(certain_points)
                        if np.any(points_inside):
                            is_clean = False
                            print(f"   ⚠️ Area {area_num} contains {np.sum(points_inside)} certain points (Drawn as red as requested).")
                    
                    # Draw the "Uncertainty Island" (All areas red as requested)
                    plt.fill(hull_vertices[:, 0], hull_vertices[:, 1], 
                             'salmon', alpha=0.15, 
                             label='Uncertainty Island' if areas_drawn == 0 else "")
                    plt.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
                             np.append(hull_vertices[:, 1], hull_vertices[0, 1]),
                             'r--', alpha=0.4, linewidth=1)
                    
                    # Add a text label matching the JSON for ALL areas found
                    center = np.mean(hull_vertices, axis=0)
                    avg_sc = np.mean([scores_list[i] for i in cluster_global_indices])
                    
                    plt.text(center[0], center[1], f"Area {area_num}\nSC: {avg_sc:.2f}", 
                             color='darkred', fontsize=9, fontweight='bold',
                             ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='r', pad=1, boxstyle='round,pad=0.2'))
                    
                    areas_drawn += 1
                except Exception as e:
                    print(f"   ⚠️ Error processing cluster {cluster_id}: {e}")
                    continue
        
        if areas_drawn == 0:
            print("💡 No clean uncertainty islands found to draw.")
            plt.scatter(mid_points[:, 0], mid_points[:, 1], s=120, 
                        facecolors='none', edgecolors='salmon', linewidths=1.2, 
                        alpha=0.4, label='Uncertain Points (Scattered)')
        
        if areas_drawn == 0:
            # If no dense clean areas found, just highlight individual uncertain points
            plt.scatter(mid_points[:, 0], mid_points[:, 1], s=120, 
                        facecolors='none', edgecolors='r', linewidths=1.5, 
                        alpha=0.6, label='Individual Uncertain Points')
    elif len(mid_points) > 0:
        plt.scatter(mid_points[:, 0], mid_points[:, 1], s=120, 
                    facecolors='none', edgecolors='r', linewidths=1.5, 
                    alpha=0.6, label='Individual Uncertain Points')
    
    plt.legend(loc='upper right')

    # 6. Save result
    output_file = os.path.join(output_dir, "question_diversity.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Diversity plot saved to: {output_file}")

if __name__ == "__main__":
    plot_question_diversity()
