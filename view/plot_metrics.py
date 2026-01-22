import os
import json
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_metrics():
    # 1. Gather all metrics files
    # Assuming the script is run from the project root or the view folder
    # We only look inside the dedicated 'metric' folder
    files = (glob.glob("metric/metrics_step_*.json") + 
             glob.glob("../metric/metrics_step_*.json"))
    
    if not files:
        print("No metrics files found. Searching for 'metrics_step_*.json' exclusively in metric/ or ../metric/")
        return

    # 2. Extract step numbers and sort files
    file_data = []
    for f in files:
        match = re.search(r"metrics_step_(\d+)\.json", f)
        if match:
            step = int(match.group(1))
            file_data.append((step, f))
    
    file_data.sort() # Sort by step number

    if not file_data:
        print("No valid metrics files found.")
        return

    print(f"Found {len(file_data)} metric files. Processing...")

    # 3. Aggregate metrics
    metrics_history = defaultdict(list)
    steps = []

    for step, filepath in file_data:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                steps.append(step)
                # Ensure all keys exist in history, even if missing in some steps
                all_keys = set(metrics_history.keys()) | set(data.keys())
                for key in all_keys:
                    val = data.get(key, None)
                    # If val is a dict or something non-plottable, skip it for now
                    if isinstance(val, (int, float)):
                        metrics_history[key].append(val)
                    else:
                        # Append None or skip? Let's skip non-numeric to avoid plotting issues
                        pass
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # 4. Create output directory for plots
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 5. Generate plots
    for key, values in metrics_history.items():
        if not values:
            continue
            
        # Re-align values with steps in case some steps were skipped due to non-numeric
        # This is simple if we just use the length of values
        current_steps = steps[:len(values)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(current_steps, values, marker='.', linestyle='-', color='b')
        plt.title(f"Metric: {key}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Clean filename: replace / and other chars
        safe_key = key.replace("/", "_").replace(".", "_")
        filename = os.path.join(output_dir, f"{safe_key}.png")
        
        plt.savefig(filename)
        plt.close()
        print(f"Generated: {filename}")

if __name__ == "__main__":
    plot_metrics()
