import os

from dotenv import load_dotenv
from openrouter import OpenRouter

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
USER_TABLE = os.getenv("USER_TABLE", "")
FEEDBACK_TABLE = os.getenv("FEEDBACK_TABLE", "")

# Two different URLs, and mixing them up sends readers to a dead route.
# *_FUNCTION_URL is where an emailed link points -- the edge function, which
# verifies the signature and does the work. *_BASE_URL is read by those
# functions to build the /success and /error pages they redirect to, and is
# not used by this process.
FEEDBACK_FUNCTION_URL = os.getenv("FEEDBACK_FUNCTION_URL", "")
UNSUBSCRIBE_FUNCTION_URL = os.getenv("UNSUBSCRIBE_FUNCTION_URL", "")
LINK_SIGNING_SECRET = os.getenv("LINK_SIGNING_SECRET", "")

LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "")

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "")
PUBMED_TOOL_NAME = os.getenv("PUBMED_TOOL_NAME", "")

EMAIL_API_KEY = os.getenv("EMAIL_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")


db_connection_string = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

llm = OpenRouter(api_key=LLM_API_KEY)
