import asyncio
import json
import struct
import logging
import os
from typing import Optional
from .agent import MathAgent

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MathAgentTCPServer")

class AsyncTCPServer:
    def __init__(self, host: str, port: int, agent: MathAgent):
        self.host = host
        self.port = port
        self.agent = agent

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {addr}")

        try:
            while True:
                # Read 4 bytes length header (Big Endian)
                header_data = await reader.read(4)
                if not header_data:
                    break  # Connection closed cleanly

                if len(header_data) < 4:
                    logger.warning(f"Incomplete header from {addr}")
                    break
                
                msg_len = struct.unpack('>I', header_data)[0]
                
                # Read the JSON body
                body_data = await reader.readexactly(msg_len)
                request_json = json.loads(body_data.decode('utf-8'))
                
                logger.info(f"Received request: {request_json.keys()}")
                
                # Process the request
                problem = request_json.get("problem")
                max_turns = request_json.get("max_turns", 5) # Default will be overridden by client
                
                if not problem:
                    response_data = {"error": "Missing 'problem' field"}
                else:
                    try:
                        # Blocking call needs to be run in thread pool if not fully async
                        # But agent.solve uses asyncio logic via OpenAI Async? 
                        # Wait, agent.py currently uses Sync OpenAI client?
                        # Let's check agent.py again.
                        # It uses self.client = OpenAI(...) which is SYNC.
                        # So we must run it in executor to avoid blocking the loop.
                        loop = asyncio.get_running_loop()
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.agent.solve(problem, max_turns=max_turns)
                        )
                        response_data = result
                    except Exception as e:
                        logger.error(f"Error solving problem: {e}", exc_info=True)
                        response_data = {"error": str(e)}

                # Serialize response
                response_bytes = json.dumps(response_data).encode('utf-8')
                res_len = len(response_bytes)
                
                # Send 4 bytes length + body
                writer.write(struct.pack('>I', res_len))
                writer.write(response_bytes)
                await writer.drain()

        except asyncio.IncompleteReadError:
            logger.warning(f"Client {addr} disconnected during read.")
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            logger.info(f"Closing connection from {addr}")
            writer.close()
            await writer.wait_closed()

    async def serve(self):
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        addr = server.sockets[0].getsockname()
        logger.info(f"Serving TCP on {addr}")

        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    # Initialize Agent
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        logger.error("OPENAI_API_KEY not set")
        exit(1)
        
    MODEL = os.getenv("MODEL_NAME", "qwen-flash-us")
    SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")
    
    agent = MathAgent(
        api_key=API_KEY, 
        model=MODEL,
        sandbox_url=SANDBOX_URL
    )
    
    # Start Server
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", 8000))
    
    tcp_server = AsyncTCPServer(HOST, PORT, agent)
    try:
        asyncio.run(tcp_server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
