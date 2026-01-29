import asyncio
import os
import logging
from .tcp_server import AsyncTCPServer
from .agent import MathAgent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MathAgentServer")

if __name__ == "__main__":
    # Environment Variables
    API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not API_KEY:
        logger.error("OPENAI_API_KEY (or LLM_API_KEY) not set in environment.")
        exit(1)
        
    MODEL = os.getenv("MODEL_NAME", "qwen-flash")
    SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080")
    PORT = int(os.getenv("PORT", 8000))
    HOST = "0.0.0.0"

    logger.info(f"Starting Math Agent TCP Server on {HOST}:{PORT}")
    logger.info(f"Model: {MODEL}, Sandbox: {SANDBOX_URL}")

    # Initialize Agent
    agent = MathAgent(
        api_key=API_KEY, 
        model=MODEL,
        sandbox_url=SANDBOX_URL
    )
    
    # Start TCP Server
    tcp_server = AsyncTCPServer(HOST, PORT, agent)
    try:
        asyncio.run(tcp_server.serve())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
