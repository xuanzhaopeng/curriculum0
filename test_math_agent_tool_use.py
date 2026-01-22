import os
import logging
from math_agent.agent import MathAgent

# Set your API key here
API_KEY = os.getenv("GEMINI_API_KEY")

def test_tool_use():
    logging.basicConfig(level=logging.INFO)
    agent = MathAgent(api_key=API_KEY)
    
    # A problem that strongly encourages tool use (simulating a complex calculation)
    problem = "Calculate the sum of the first 100 prime numbers."
    
    print(f"Testing problem: {problem}")
    result = agent.solve(problem)
    
    print("\n--- RESULTS ---")
    print(f"Final Answer: {result.get('final_answer')}")
    print(f"Tool Calls Count: {result.get('tool_calls')}")
    print(f"Reasoning snippet: {result.get('raw_reasoning')[:500]}...")

if __name__ == "__main__":
    if not API_KEY:
        print("Please set GEMINI_API_KEY environment variable.")
    else:
        test_tool_use()
