import socket
import json
import struct
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def call_math_agent_tcp(problem, host="localhost", port=8000, max_turns=5):
    """Call the Math Agent TCP service with a single problem."""
    payload = {
        "problem": problem,
        "max_turns": max_turns,
        "model": "gemini-2.0-flash"
    }
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        # Serialize
        req_bytes = json.dumps(payload).encode('utf-8')
        req_len = len(req_bytes)
        
        # Send
        sock.sendall(struct.pack('>I', req_len))
        sock.sendall(req_bytes)
        
        # Receive Header
        header_data = sock.recv(4)
        if not header_data:
            return {"error": "No data received", "problem": problem}
            
        resp_len = struct.unpack('>I', header_data)[0]
        
        # Receive Body
        body_data = b""
        while len(body_data) < resp_len:
            packet = sock.recv(resp_len - len(body_data))
            if not packet:
                break
            body_data += packet
            
        response = json.loads(body_data.decode('utf-8'))
        response["problem"] = problem
        
        sock.close()
        return response
        
    except Exception as e:
        return {"error": str(e), "problem": problem}

def test_parallel_math_agent():
    """Test the Math Agent with 10 different questions in parallel."""
    url = os.getenv("MATH_AGENT_URL", "localhost:8000")
    if ":" in url:
        host, port_str = url.replace("tcp://", "").replace("http://", "").split(":")
        port = int(port_str)
    else:
        host = "localhost"
        port = 8000

    # 10 very complex math problems that REQUIRE Python tool usage
    problems = [
        "Find the sum of all prime numbers less than 10000.",
        "Calculate the 1000th Fibonacci number modulo 1000000007.",
        "What is the sum of all proper divisors of 1000000? (A proper divisor is a divisor less than the number itself)",
        "How many prime numbers are there between 1 and 100000?",
        "Calculate the determinant of a 5x5 matrix where entry (i,j) = i*j + i + j.",
        "Find the smallest positive integer n such that n! ends with exactly 100 zeros.",
        "Calculate the sum of the first 1000 terms of the series: 1/1^2 + 1/2^2 + 1/3^2 + ... (approximation of π²/6)",
        "What is the 10000th digit after the decimal point in the value of π?",
        "Find the number of integer solutions to x² + y² + z² = 1000 where x, y, z are non-negative.",
        "Calculate the expected value of the maximum when rolling 10 fair six-sided dice (simulate with 100000 trials)."
    ]
    
    print(f"Testing Math Agent at {host}:{port} with {len(problems)} questions in parallel...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(call_math_agent_tcp, p, host, port): i for i, p in enumerate(problems)}
        
        results = []
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Error processing question {idx}: {e}")
                results.append((idx, {"error": str(e), "problem": problems[idx]}))
    
    # Sort by original index
    results.sort(key=lambda x: x[0])
    
    end_time = time.time()
    
    # Print results
    print(f"\nCompleted {len(results)} questions in {end_time - start_time:.2f} seconds")
    print("=" * 80)
    
    for idx, result in results:
        print(f"\n[Question {idx + 1}] {result.get('problem', 'Unknown')}")
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Answer: {result.get('final_answer', 'N/A')}")
            print(f"  🔧 Tool Calls: {result.get('tool_calls', 0)}")
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Total: {len(results)}")
    print(f"  Success: {len([r for _, r in results if 'error' not in r])}")
    print(f"  Failed: {len([r for _, r in results if 'error' in r])}")
    print(f"  Avg time: {(end_time - start_time) / len(results):.2f}s per question")

if __name__ == "__main__":
    test_parallel_math_agent()
