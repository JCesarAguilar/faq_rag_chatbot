import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

MIN_CHUNK_WORDS = 30
TOP_K = 2  # Entre 2 y 5

INDEX_PATH = "data/index.json"
PROMPT_PATH = "prompts/main_prompt.txt"
