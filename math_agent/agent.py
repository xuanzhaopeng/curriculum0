import re
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .sandbox_fusion import SandboxFusionClient

logger = logging.getLogger(__name__)

class MathAgent:
    """
    A math agent that solves problems using Planning, Chain of Thought, 
    and optional Python execution via SandboxFusion.
    """
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.5-flash",
        sandbox_url: str = "http://localhost:8080"
    ):
        """
        Initialize the MathAgent.
        
        Args:
            api_key: OpenAI-compatible API key.
            base_url: OpenAI-compatible base URL.
            model: Model name to use.
            sandbox_url: The base URL for the SandboxFusion service.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.sandbox = SandboxFusionClient(base_url=sandbox_url)
        
        self.system_prompt = (
            "You are a highly capable Math Agent. Your goal is to solve math problems with extreme accuracy.\n\n"
            "Follow this structured approach for every problem:\n"
            "1. **Plan**: Analyze the problem and describe your step-by-step strategy to solve it.\n"
            "2. **Chain of Thought**: Execute your plan. Show all your reasoning and intermediate calculations.\n"
            "3. **Tool Use (Optional)**: If you need to perform complex calculations, simulations, or verify "
            "mathematical properties, you can use a Python interpreter. Wrap your code in <python> and </python> tags. "
            "You will receive the output in the next turn.\n"
            "4. **Final Answer**: Once you have the solution, provide the final answer clearly. Wrap the numeric "
            "result in \\boxed{} for clarity.\n\n"
            "### Example Format:\n"
            "**Plan**\n"
            "I will first calculate X, then Y...\n\n"
            "**Chain of Thought**\n"
            "Step 1: ...\n"
            "Step 2: ...\n\n"
            "**Final Answer**\n"
            "The answer is \\boxed{42}\n\n"
            "Rules:\n"
            "- Always include 'Plan', 'Chain of Thought', and 'Final Answer' headers.\n"
            "- The final result MUST be inside \\boxed{}.\n"
            "- Call Python ONLY when necessary for accuracy or efficiency.\n"
        )

    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Helper to get completion from LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1, # Lower temperature for more consistent formatting
        )
        return response.choices[0].message.content

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """Extract the content inside \boxed{}."""
        pattern = r"\\boxed\{(.*?)\}"
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip() # Return the last boxed answer found
        return None

    def solve(self, problem: str, max_turns: int = 5) -> Dict[str, Any]:
        """
        Solve a math problem using a multi-turn reasoning loop.
        Returns a dict with 'raw_reasoning' and 'final_answer'.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem}"}
        ]
        
        final_response = ""
        full_history = []
        
        for turn in range(max_turns):
            logger.info(f"MathAgent Turn {turn + 1}")
            response = self._generate_response(messages)
            messages.append({"role": "assistant", "content": response})
            full_history.append(response)
            final_response = response
            
            # Extract python code if present
            code_match = re.search(r"<python>(.*?)</python>", response, re.DOTALL)
            if not code_match:
                # If no tool use, we check if there is a boxed answer
                if self._extract_final_answer(response):
                    break # Finished normally
                else:
                    # If no boxed answer, we try one more time to ask for it specifically
                    # but only if we haven't already tried to recover in this loop
                    if turn < max_turns - 1:
                        logger.info("No boxed answer found, prompt for final answer.")
                        messages.append({"role": "user", "content": "Please conclude with your final answer in \\boxed{}"})
                        continue
                    break
            
            code = code_match.group(1).strip()
            logger.info(f"Executing Python code:\n{code}")
            
            execution_result = self.sandbox.run_python(code)
            
            run_result = execution_result.get("run_result", {})
            stdout = run_result.get("stdout", "")
            stderr = run_result.get("stderr", "")
            
            if not stdout and not stderr:
                stdout = execution_result.get("stdout", "")
                stderr = execution_result.get("stderr", "")
            
            observation = f"Observation (Python Output):\n{stdout}"
            if stderr:
                observation += f"\nErrors:\n{stderr}"
                
            messages.append({"role": "user", "content": observation})
            logger.info("Observation received and sent back to agent.")

        # Final check: if still no boxed answer, do a final forceful extraction attempt
        answer = self._extract_final_answer(final_response)
        if answer is None:
            logger.info("Last resort: asking for boxed answer specifically.")
            messages.append({"role": "user", "content": "Please provide the final numeric answer inside \\boxed{} now. No other text."})
            recovery_response = self._generate_response(messages)
            answer = self._extract_final_answer(recovery_response)
            if answer:
                final_response += "\n\n**Recovery**\n" + recovery_response

        return {
            "raw_reasoning": final_response,
            "final_answer": answer
        }

if __name__ == "__main__":
    # Example usage with environment variables
    import os
    logging.basicConfig(level=logging.INFO)
    
    API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key")
    agent = MathAgent(api_key=API_KEY)
    
    # test_problem = "What is the 10th Fibonacci number?"
    # result = agent.solve(test_problem)
    # print(f"\nFinal Result:\n{result}")
