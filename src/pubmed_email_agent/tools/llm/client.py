from typing import cast, Any, List

import json
from pubmedclient.models import ESearchRequest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.pubmed_email_agent.agent.state import Summary
from src.pubmed_email_agent.tools.user.client import UserProfile
from src.pubmed_email_agent.prompts import (
    GENERATE_QUERY_SYS,
    GENERATE_QUERY_USER,
    SUMMARIZE_ARTICLE_SYS,
    SUMMARIZE_ARTICLE_USER,
    FORMAT_EMAIL_SYS,
    FORMAT_EMAIL_USER,
)


class LLMTools:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.search_llm = self.llm.with_structured_output(ESearchRequest)

    def generate_search_query(
        self, interests: list[str], search_from_date: str, article_count: int = 5
    ) -> ESearchRequest:
        """
        Generates PubMed search parameters
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERATE_QUERY_SYS),
                ("user", GENERATE_QUERY_USER),
            ]
        )

        chain = prompt | self.search_llm

        response = chain.invoke(
            {
                "interests": ", ".join(interests),
                "date": search_from_date,
                "article_count": article_count,
            }
        )

        response = cast(ESearchRequest, response)

        return response

    def summarize_article(self, article_data: dict[str, Any]) -> str:
        """
        Summarizes the given article data
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARIZE_ARTICLE_SYS),
                ("user", SUMMARIZE_ARTICLE_USER),
            ]
        )

        chain = prompt | self.llm

        response = chain.invoke({"article_data": article_data})

        return response.text()

    def format_email(self, user_profile: UserProfile, summaries: List[Summary]) -> str:
        """
        Formats the email content based on the summary
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FORMAT_EMAIL_SYS),
                ("user", FORMAT_EMAIL_USER),
            ]
        )

        chain = prompt | self.llm

        response = chain.invoke(
            {
                "first_name": user_profile.first_name,
                "user_profile": str(user_profile),
                "summaries_json": json.dumps([dict(s) for s in summaries], indent=2),
            }
        )

        return response.text()
