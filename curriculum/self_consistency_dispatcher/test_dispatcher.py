import requests
import json
import time

def test_dispatcher_service():
    url = "http://localhost:8001/dispatch"
    
    payload = {
        "question": "If I have 3 apples and you give me 5 more, how many do I have total?",
        "n": 3,  # Small n for quick test
        "max_turns": 3
    }

    print(f"Sending dispatch request to {url}...")
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        end_time = time.time()
        
        result = response.json()
        print(f"\n--- Dispatcher Response (Time: {end_time - start_time:.2f}s) ---")
        print(f"Question: {result['question']}")
        print(f"Majority Answer: {result['majority_answer']}")
        print(f"Self-Consistency Score: {result['self_consistency_score']:.2f} ({result['total_samples']} samples)")
        print(f"All Answers: {result['all_answers']}")
        print("------------------------------------------\n")
        
        if result['self_consistency_score'] > 0:
            print("SUCCESS: Dispatcher successfully aggregated answers.")
        else:
            print("WARNING: Dispatcher returned 0 consistency. Check Math Agent logs.")
            
    except Exception as e:
        print(f"ERROR: Failed to call Dispatcher service. {e}")

if __name__ == "__main__":
    test_dispatcher_service()
