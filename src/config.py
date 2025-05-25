import os
from dotenv import load_dotenv

load_dotenv()

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Milvus Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "sever_meglog_collection")
SIMILARITY_TOP_K = 20

# Bot Configuration
DISCORD_ADMIN_ID = os.getenv("DISCORD_ADMIN_ID", "admin_ID")  # Default to placeholder

# Embedding Model
EMBEDDING_MODEL_NAME = "cl-nagoya/ruri-v3-310m"
EMBEDDING_MODEL_DIMENSION = 768 # Dimension for cl-nagoya/ruri-v3-310m

# LLM Model (Gemini)
LLM_MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Default to Gemini 2.5

# LlamaIndex Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

# LLM Temperature for generation (0.0-1.0)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# File to store crawled guild IDs
CRAWLED_GUILDS_FILE = "crawled_guilds.json"

# Data Collection Settings
MAX_MESSAGES_PER_CHANNEL = None # 各チャンネルから収集する最大メッセージ数 (Noneで無制限)

if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable not set.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.") 