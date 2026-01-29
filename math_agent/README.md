# Math Agent

A powerful math-solving agent that uses **Planning**, **Chain of Thought (CoT)**, and **Python Tool Execution** to solve complex mathematical problems. It is built to work with OpenAI-compatible APIs (defaults to Google Gemini).

## Features
- **Planning & Reasoning**: Every solution starts with a structured plan and detailed reasoning steps.
- **Python Sandbox**: Automatically executes Python code via SandboxFusion for calculations and verification.
- **OpenAI Protocol**: Fully compatible with Gemini (default), OpenAI, and other standard LLM endpoints.
- **HTTP Interface**: Exposes a simple REST API for integration.

## Setup with Conda

1. **Create and activate a new conda environment:**
   ```bash
   conda create -n math-agent python=3.10
   conda activate math-agent
   ```

2. **Install dependencies:**
   Navigate to the `math_agent` folder and install the requirements:
   ```bash
   pip install -r math_agent/requirements.txt
   ```

## Starting the Service

1. **Configure Environment Variables:**
   Set your LLM API key (e.g., Gemini API Key):
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

2. **Launch the Server:**
   Run the server from the root of the repository to ensure package imports work correctly:
   ```bash
   # From the repository root (Agent0-curriculum)
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python -m math_agent.server
   ```
   The service will start on `http://0.0.0.0:8000`.

## Starting SandboxFusion (Docker)

The Math Agent requires the SandboxFusion service to execute Python code. Start it using the specific server image tag:

```bash
docker run -d \
  --name sandbox-fusion \
  -p 8080:8080 \
  volcengine/sandbox-fusion:server-20250609
```

Wait a few seconds for the container to initialize. You can check the logs to ensure it is running:
```bash
docker logs -f sandbox-fusion
```

### Verification
You can verify the sandbox is working by running a simple Python test:
```bash
curl -s -X POST http://localhost:8080/run_code \
     -H "Content-Type: application/json" \
     -d '{"code": "print(1+1)", "language": "python"}'
```
You should see a JSON response stating `"stdout":"2\n"`.

> **Note**: The `latest` tag is often unavailable on Docker Hub for this repository; always prefer the `server-20250609` tag or check for the newest specific release.

## How it Works (`agent.py` Analysis)

The Math Agent follows a **Multi-turn Re-Act (Reasoning and Acting)** pattern:
1. **System Prompt**: Enforces a strict structure: `Plan` -> `Chain of Thought` -> `Tool Use (Optional)` -> `Final Answer`.
2. **Loop**: The agent runs up to `max_turns` (default 5). If it provides a Python code block, the internal loop pauses, executes the code, and feeds the output back as an `Observation`.
3. **State Management**: The entire conversation history (including plans, code, and observations) is maintained to ensure the agent understands the context of its previous actions.

## Example Response

When you call `/solve`, the response contains both the full reasoning trace and the extracted numeric result.

**Example JSON Response:**
```json
{
  "problem": "What is the 5th prime number?",
  "raw_reasoning": "1. **Plan**: Count primes starting from 2...\n2. **Chain of Thought**: Primes: 2, 3, 5, 7, 11.\n3. **Final Answer**: The 5th prime is \\boxed{11}",
  "final_answer": "11"
}
```

- `raw_reasoning`: The complete output from the agent including Planning, CoT, and `\boxed{}` markers.
- `final_answer`: The content extracted specifically from the `\boxed{}` tag.

## How to Call the Agent

### Using `curl`
You can send a POST request to the `/solve` endpoint.

```bash
curl -X POST "http://localhost:8000/solve" \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $OPENAI_API_KEY" \
     -d '{
       "problem": "Find the sum of all prime numbers between 1 and 50.",
       "model": "qwen-flash",
       "max_turns": 5
     }'
```

### Request Parameters
| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `problem` | String | The math question to solve. | (Required) |
| `max_turns` | Integer | Max reasoning/tool-use loops. | `5` |
| `sandbox_url` | String | URL of the SandboxFusion service. | `http://localhost:8080` |
| `model` | String | LLM model name. | `qwen-flash` |
| `base_url` | String | OpenAI-compatible base URL. | Gemini OpenAI Endpoint |

### API Endpoints
- `POST /solve`: Submit a problem for the agent to solve.
- `GET /health`: Check if the service is running.
