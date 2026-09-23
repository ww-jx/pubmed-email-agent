import asyncio

from src.pubmed_email_agent.agent.graph import Agent
from src.pubmed_email_agent.config import (
    EMAIL_API_KEY,
    FEEDBACK_FUNCTION_URL,
    FEEDBACK_TABLE,
    FROM_EMAIL,
    LINK_SIGNING_SECRET,
    LLM_MODEL,
    PUBMED_EMAIL,
    PUBMED_TOOL_NAME,
    UNSUBSCRIBE_FUNCTION_URL,
    USER_TABLE,
    db_connection_string,
    llm,
)
from src.pubmed_email_agent.logger import get_logger
from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools
from src.pubmed_email_agent.tools.user.client import UserTools

logger = get_logger(__name__)


async def main():
    user_tools = UserTools(
        db_connection_string, USER_TABLE, FEEDBACK_TABLE, EMAIL_API_KEY, FROM_EMAIL
    )
    llm_tools = LLMTools(
        client=llm,
        model=LLM_MODEL,
        feedback_function_url=FEEDBACK_FUNCTION_URL,
        unsubscribe_function_url=UNSUBSCRIBE_FUNCTION_URL,
        link_signing_secret=LINK_SIGNING_SECRET,
    )
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
