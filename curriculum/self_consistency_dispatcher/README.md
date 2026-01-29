# Self-Consistency Dispatcher

This service implements the Self-Consistency mechanism described in the [Agent0 paper](https://arxiv.org/pdf/2511.16043). It acts as a middleware that samples multiple solutions from the Math Agent and calculates the majority-voted answer and its confidence score.

## How it Works
1. Receives a question and a sample size $n$.
2. Concurrently dispatches $n$ requests to the Math Agent.
3. Collects all final answers.
4. Calculates the self-consistency score: $p(x) = \frac{\text{count}(\text{majority\_answer})}{n}$.

## Setup

1. Activate your conda environment.
2. Install dependencies:
   ```bash
   pip install -r curriculum/self_consistency_dispatcher/requirements.txt
   ```

## Starting the Service

Ensure the **Math Agent** service is already running on port 8000.

Run the dispatcher from the repository root:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m curriculum.self_consistency_dispatcher.server
```
The service will start on `http://0.0.0.0:8001`.

## API Usage

### `POST /dispatch`
**Payload:**
```json
{
  "question": "What is 2+2?",
  "n": 10,
  "max_turns": 5,
  "model": "qwen-flash-us"
}
```

**Response:**
```json
{
  "question": "What is 2+2?",
  "majority_answer": "4",
  "self_consistency_score": 1.0,
  "total_samples": 10,
  "all_answers": ["4", "4", ...],
  "raw_responses": [...]
}
```
