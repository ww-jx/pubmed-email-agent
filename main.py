import asyncio

from src.pubmed_email_agent.agent.graph import Agent
from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.user.client import UserTools
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools

from src.pubmed_email_agent.config import (
    DB_TABLE,
    EMAIL_API_KEY,
    FROM_EMAIL,
    PUBMED_EMAIL,
    PUBMED_TOOL_NAME,
    db_connection_string,
    llm,
)


async def main():
    user_tools = UserTools(db_connection_string, DB_TABLE, EMAIL_API_KEY, FROM_EMAIL)
    llm_tools = LLMTools(llm)
    pubmed_tools = PubmedTools(PUBMED_TOOL_NAME, PUBMED_EMAIL)

    agent = Agent(user_tools, llm_tools, pubmed_tools)

    user_ids = await user_tools.get_all_user_ids()

    for user_id in user_ids:
        try:
            initial_state = {
                "user_id": user_id,
            }

            await agent.run(initial_state)
            print(f"Processed user {user_id} successfully.")
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue

    print("All users processed.")


if __name__ == "__main__":
    asyncio.run(main())
