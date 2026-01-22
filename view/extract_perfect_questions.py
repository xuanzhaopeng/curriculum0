import os
import json
import glob
from tqdm import tqdm

def extract_perfect_questions():
    # 1. Gather all question files
    # Search in ./questions/ and ../questions/
    files = glob.glob("questions/results_sc_*.json") + glob.glob("../questions/results_sc_*.json")
    
    if not files:
        print("No question files found in ./questions/ or ../questions/")
        return

    # 2. Extract entries with score 1.0
    perfect_questions = {} # Using dict to deduplicate by question text
    
    print(f"Processing {len(files)} files...")
    for f in tqdm(files):
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                for entry in data:
                    q = entry.get("question", "").strip()
                    score = entry.get("self_consistency_score", 0.0)
                    
                    if q and float(score) >= 0.9999: # Handling potential float precision
                        # Store the full entry
                        perfect_questions[q] = entry
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 3. Save to output
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results_list = list(perfect_questions.values())
    output_file = os.path.join(output_dir, "perfect_questions.json")
    
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"✅ Extracted {len(results_list)} perfect questions.")
    print(f"✅ Saved to: {output_file}")

if __name__ == "__main__":
    extract_perfect_questions()
