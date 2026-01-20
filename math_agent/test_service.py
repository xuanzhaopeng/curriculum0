import socket
import json
import struct
import os

def test_math_agent_tcp():
    url = os.getenv("MATH_AGENT_URL", "localhost:8000")
    if ":" in url:
        host, port_str = url.replace("tcp://", "").replace("http://", "").split(":")
        port = int(port_str)
    else:
        host = "localhost"
        port = 8000

    print(f"Connecting to {host}:{port} via TCP...")

    payload = {
        "problem": "Calculate the sum of primes between 10 and 20.",
        "max_turns": 4,
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
            print("Error: No data received")
            return
            
        resp_len = struct.unpack('>I', header_data)[0]
        
        # Receive Body
        body_data = b""
        while len(body_data) < resp_len:
            packet = sock.recv(resp_len - len(body_data))
            if not packet:
                break
            body_data += packet
            
        response = json.loads(body_data.decode('utf-8'))
        
        print("\nParsed Response:")
        print(f"Final Answer: {response.get('final_answer')}")
        print("-" * 40)
        print(f"Raw Reasoning:\n{response.get('raw_reasoning')}")
        
        sock.close()
        
    except Exception as e:
        print(f"Error testing service: {e}")

if __name__ == "__main__":
    test_math_agent_tcp()
