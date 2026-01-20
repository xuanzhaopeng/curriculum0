
import os
import logging
from math_agent.agent import MathAgent

logging.basicConfig(level=logging.INFO)

def test_recovery():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY")
        return

    agent = MathAgent(api_key=api_key)
    
    # We want to see if it handles a case where it might forget the boxed answer.
    # We can't easily force the model to forget, but we can check the logs
    # and see if our code handles the logic correctly.
    
    problem = "What is 15 * 13? Just tell me the number in words first, then later I will ask for the boxed answer if you forget."
    # We'll set max_turns to 2 to see the recovery turn in action if it doesn't box it.
    
    print("\n--- Testing MathAgent solve with recovery logic ---")
    result = agent.solve(problem, max_turns=2)
    
    print(f"\nFinal Answer: {result['final_answer']}")
    print(f"Raw Reasoning Snippet:\n{result['raw_reasoning'][-300:]}")
    
    if result['final_answer']:
        print("\nSUCCESS: Found a boxed answer.")
    else:
        print("\nFAILURE: No boxed answer found even with recovery.")

if __name__ == "__main__":
    test_recovery()
