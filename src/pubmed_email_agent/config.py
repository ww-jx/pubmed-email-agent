import os
from dotenv import load_dotenv
from langchain_openai.chat_models.base import BaseChatOpenAI

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_TABLE = os.getenv("DB_TABLE", "")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_API_BASE = os.getenv("OPENROUTER_BASE_URL")
LLM_MODEL = os.getenv("OPENROUTER_MODEL")

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
PUBMED_TOOL_NAME = os.getenv("PUBMED_TOOL_NAME", "")

EMAIL_API_KEY = os.getenv("EMAIL_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")


db_connection_string = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

llm = BaseChatOpenAI(
    openai_api_key=LLM_API_KEY,
    openai_api_base=LLM_API_BASE,
    model=LLM_MODEL,
)
