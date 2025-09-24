from typing import cast, Any, List

import json
from pubmedclient.models import ESearchRequest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from src.pubmed_email_agent.agent.state import Summary
from src.pubmed_email_agent.tools.user.client import UserProfile


class LLMTools:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        self.search_llm = self.llm.with_structured_output(ESearchRequest)

    def generate_search_query(
        self, interests: list[str], search_from_date: str, article_count: int = 3
    ) -> ESearchRequest:
        """
        Generates PubMed search parameters
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that helps to generate PubMed search queries."
                    "Based on the user's interests and preferences, generate the search parameters",
                ),
                (
                    "user",
                    "Generate the search parameters based on the following details: \n"
                    "- Interests: {interests}\n"
                    "- Only retrieve results published after {date}\n"
                    "- The query should only return at most {article_count} articles",
                ),
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
                (
                    "system",
                    "You are a helpful assistant that helps to summarize PubMed articles."
                    "Based on the article data, generate a concise summary that is easy to understand.",
                ),
                (
                    "user",
                    "Summarize the following article data: \n" "{article_data}\n",
                ),
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
                (
                    "system",
                    "You are an expert medical writer. Your task is to create a personalized weekly email summary "
                    "of recent clinical research for a user. The tone should be friendly, informative, and clear. "
                    "The output must be in well-structured Markdown format. "
                    "Always include a disclaimer that this is not medical advice.",
                ),
                (
                    "user",
                    "Please create the email using the following information:\n\n"
                    "**User Profile:**\n"
                    "{user_profile}\n\n"
                    "**Summarized Articles:**\n"
                    "{summaries_json}\n\n"
                    "**Email Structure Requirements:**\n"
                    "1.  **Main Title:** Start with '{first_name}'s Evidence Weekly'.\n"
                    "2.  **TL;DR Section:** Analyze all the provided articles and create a high-level, one-line summary for each of the user's conditions.\n"
                    "3.  **Detailed Article Sections:** Create a separate, dedicated section for EACH article provided in the summaries. Each section must include the article's full Title, its PMID, its Link, and its full Summary.\n"
                    "4.  **Analysis & PCP Questions:** After the article sections, create general sections for 'Prep for Your PCP' and 'Smart Questions to Ask', based on the user's conditions and the findings from the articles.\n"
                    "5.  **Link Index:** Create a final, clean list of all PubMed links.",
                ),
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
