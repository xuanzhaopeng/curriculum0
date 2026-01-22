import requests
import json
from typing import Dict, Any, Optional

class SandboxFusionClient:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')

    def run_python(self, code: str, compile_timeout: int = 30, run_timeout: int = 30) -> Dict[str, Any]:
        """
        Executes Python code using SandboxFusion.
        """
        url = f"{self.base_url}/run_code"
        payload = {
            "code": code,
            "language": "python",
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout
        }
        
        try:
            response = requests.post(url, json=payload, timeout=run_timeout + compile_timeout + 5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌❌❌ Error running code: {e}")
            return {
                "status": "error",
                "message": str(e),
                "stdout": "",
                "stderr": str(e)
            }

if __name__ == "__main__":
    # Quick test if SandboxFusion is running locally
    client = SandboxFusionClient()
    result = client.run_python("print(1 + 1)")
    print(result)
