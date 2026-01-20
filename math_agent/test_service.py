import requests
import json
import os

def test_math_agent_service():
    url = "http://localhost:8000/solve"
    
    # We can pass the API key via Header or let the server use its env var
    api_key = os.getenv("GEMINI_API_KEY")
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {
        "problem": "Calculate the sum of primes between 10 and 20.",
        "max_turns": 5,
        "model": "gemini-2.5-flash"
    }

    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("\n--- Response Received ---")
        print(f"Problem: {result['problem']}")
        print(f"Final Answer: {result['final_answer']}")
        print(f"Raw Reasoning:\n{result['raw_reasoning']}")
        print("-------------------------\n")
        
        if result['final_answer']:
            print(f"SUCCESS: Extracted final answer: {result['final_answer']}")
        else:
            print("WARNING: No final answer extracted. Check raw_reasoning.")
            
    except Exception as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"ERROR: Service returned: {e.response.text}")
        print(f"ERROR: Failed to call MathAgent service. {e}")

if __name__ == "__main__":
    test_math_agent_service()
