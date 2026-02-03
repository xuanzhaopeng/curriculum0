
import os
import json
import glob
import re
import math
from typing import Optional, List, Tuple
from openai import OpenAI
from jinja2 import Template

class PromptOptimizer:
    def __init__(self, 
                 prompt_dir: str = "/workspace/curriculum0/format_prompt",
                 results_dir: str = "/workspace/curriculum0/results",
                 model_name: str = "qwen-plus-latest",
                 api_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                 api_key: Optional[str] = None):
        
        self.prompt_dir = prompt_dir
        self.results_dir = results_dir
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        else:
            print("⚠️ WARNING: OPENAI_API_KEY not provided. Optimization will fail.")

    def get_latest_prompt_file(self, prefix: str = "prompt") -> Tuple[Optional[int], Optional[str]]:
        """Finds the prompt file with the highest index N in prompt_N.jinja."""
        if not os.path.exists(self.prompt_dir):
            return None, None
            
        files = glob.glob(os.path.join(self.prompt_dir, f"{prefix}_*.jinja"))
        if not files:
            return None, None
        
        candidates = []
        for f in files:
            basename = os.path.basename(f)
            m = re.match(rf"{prefix}_(\d+)\.jinja", basename)
            if m:
                candidates.append((int(m.group(1)), f))
        
        if not candidates:
            return None, None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0] # (N, filepath)

    def get_latest_summary_file(self) -> Optional[str]:
        """Finds the latest batch_summary_*.json file."""
        if not os.path.exists("questions"):
            return None
            
        files = glob.glob(os.path.join("questions", "batch_summary_*.json"))
        if not files:
            return None
        
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    def load_batch_summary(self, summary_file: str) -> str:
        """Loads the batch summary content."""
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            print(f"Error loading summary: {e}")
            return "{}"

    def optimize(self):
        """Main execution loop for a single optimization step."""
        if not self.client:
            print("❌ Cannot proceed without API client.")
            return

        print("Starting optimization loop...")
        
        # 1. Identify current prompt
        current_n, current_prompt_path = self.get_latest_prompt_file()
        if not current_prompt_path:
            print("⚠️ No existing prompt found. Please create prompt_0.jinja first.")
            return

        print(f"Current Prompt: {current_prompt_path} (Version {current_n})")
        
        with open(current_prompt_path, 'r') as f:
            current_prompt_content = f.read()

        # 2. Identify latest batch summary
        summary_file = self.get_latest_summary_file()
        if not summary_file:
            print("⚠️ No batch summary found. Cannot optimize.")
            return

        print(f"Using batch summary from: {summary_file}")
        summary_content = self.load_batch_summary(summary_file)

        # 3. Construct Meta-Prompt
        system_instruction = (
            "You are an expert meta-optimizer for LLM prompts. "
            "Your goal is to improve a system prompt used to generate high-quality reasoning tasks.\n"
            "You must DYNAMICALLY ADJUST the difficulty based on the provided QUANTITATIVE METRICS (Self-Consistency Score 'SC').\n\n"
            "### ADJUSTMENT STRATEGY:\n"
            "1. **If Avg SC > 0.7 (Too Easy)**: \n"
            "   - ACTION: INCREASE DIFFICULTY significantly. Ask for more complex steps, obscure theorems, or multi-layered logic.\n"
            "2. **If Avg SC < 0.3 (Too Hard/Ambiguous)**: \n"
            "   - ACTION: DECREASE DIFFICULTY. Simplify constraints, ensure clarity, use standard concepts.\n"
            "3. **If 0.3 <= Avg SC <= 0.7 (Optimal)**: \n"
            "   - ACTION: MAINTAIN DIFFICULTY. Focus on increasing NOVELTY and CREATIVITY to avoid repetition.\n\n"
            "### CORE REQUIREMENTS:\n"
            "1. **Format Valid**: Ensure STRICT adherence to the XML tags <question>...</question> and \\boxed{...}.\n"
            "2. **Novelty**: Avoid standard or repetitive templates.\n"
            "3. **Clarity**: Unambiguous problem statements.\n\n"
            "You will be given the CURRENT PROMPT and QUANTITATIVE METRICS.\n"
            "Analyze the metrics first to determine the direction (Harder/Easier/Same), then write the NEW PROMPT template.\n"
            "If the current prompt is performing perfectly (SC ~ 0.5, valid format, high novelty) and NO changes are needed, return ONLY the string 'NO CHANGE'."
        )
        
        user_content = (
            f"### CURRENT PROMPT OLD VERSION:\n{current_prompt_content}\n\n"
            f"### BATCH METRICS (Quantitative Feedback):\n{summary_content}\n\n"
            "### INSTRUCTIONS:\n"
            "Write the FULL content of the new prompt_N.jinja file. "
            "Keep the structure compatible (e.g. keep the {{ examples_text }} placeholder if it existed and is useful). "
            "Ensure the output is ONLY the raw text of the new prompt, no markdown backticks around it."
        )

        # 4. Call Optimizer LLM
        try:
            print("Asking LLM to optimize...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7 
            )
            new_prompt_content = response.choices[0].message.content.strip()
            
            # cleanup if markdown block
            if new_prompt_content.startswith("```"):
                lines = new_prompt_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                new_prompt_content = "\n".join(lines)

            # Check for NO CHANGE
            if not new_prompt_content.strip() or new_prompt_content.strip() == "NO CHANGE":
                print("Optimizer decided to keep the prompt unchanged.")
                new_prompt_content = current_prompt_content

            # 5. Save new prompt
            next_n = current_n + 1
            new_filename = f"prompt_{next_n}.jinja"
            new_path = os.path.join(self.prompt_dir, new_filename)
            
            with open(new_path, 'w') as f:
                f.write(new_prompt_content)
            
            print(f"✅ Successfully created new prompt: {new_path}")
            print("Preview of new prompt start:")
            print(new_prompt_content[:200] + "...")
            
            return new_path

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None
