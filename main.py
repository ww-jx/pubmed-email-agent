import asyncio

from src.pubmed_email_agent.agent.graph import Agent
from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.user.client import UserTools
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools

from src.pubmed_email_agent.config import (
    USER_TABLE,
    FEEDBACK_TABLE,
    FEEDBACK_BASE_URL,
    UNSUBSCRIBE_BASE_URL,
    EMAIL_API_KEY,
    FROM_EMAIL,
    PUBMED_EMAIL,
    PUBMED_TOOL_NAME,
    db_connection_string,
    llm,
)
from src.pubmed_email_agent.logger import get_logger

logger = get_logger(__name__)


async def main():
    user_tools = UserTools(
        db_connection_string, USER_TABLE, FEEDBACK_TABLE, EMAIL_API_KEY, FROM_EMAIL
    )
    llm_tools = LLMTools(llm, FEEDBACK_BASE_URL, UNSUBSCRIBE_BASE_URL)
    pubmed_tools = PubmedTools(PUBMED_TOOL_NAME, PUBMED_EMAIL)

    agent = Agent(user_tools, llm_tools, pubmed_tools, article_count=5, max_retries=3)

    user_ids = await user_tools.get_all_user_ids()

    for user_id in user_ids:
        try:
            initial_state = {
                "user_id": user_id,
            }

            await agent.run(initial_state)
            logger.info(f"Processed user {user_id} successfully.")
        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
            continue

    logger.info("All users processed.")


if __name__ == "__main__":
    asyncio.run(main())
