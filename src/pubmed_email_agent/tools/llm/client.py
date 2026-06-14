from typing import Any, List
from pydantic import BaseModel, Field

import json
from pubmedclient.models import ESearchRequest
from openrouter import OpenRouter
from langsmith import traceable
from fastembed import TextEmbedding

from src.pubmed_email_agent.agent.state import Summary
from src.pubmed_email_agent.tools.user.client import UserProfile
from src.pubmed_email_agent.prompts import (
    GENERATE_QUERY_SYS,
    GENERATE_QUERY_USER,
    SUMMARIZE_ARTICLE_SYS,
    SUMMARIZE_ARTICLE_USER,
    FORMAT_EMAIL_SYS,
    FORMAT_EMAIL_USER,
    EXTRACT_ARTICLE_INTENT,
)
from src.pubmed_email_agent.logger import get_logger

logger = get_logger(__name__)


class LLMPubMedQuery(BaseModel):
    term: str = Field(
        description=(
            "The exact Entrez text query string to send to PubMed. "
            "You MUST use standard PubMed syntax and Boolean operators (AND, OR, NOT). "
            "Use field tags where appropriate, such as [MeSH Terms] or [Title/Abstract]. "
            'Example: \'("Diabetes Mellitus, Type 2"[MeSH Terms]) AND ("Diet"[Title/Abstract]) NOT ("mice"[Title/Abstract])\''
        )
    )


class LLMTools:
    def __init__(
        self,
        client: OpenRouter,
        model: str,
        feedback_base_url: str,
        unsubscribe_base_url: str,
    ):
        self.client = client
        self.model = model
        self.feedback_base_url = feedback_base_url
        self.unsubscribe_base_url = unsubscribe_base_url

        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    @traceable(name="generate_search_query")
    async def generate_search_query(
        self,
        interests: list[str],
        negative_keywords: list[str],
        search_from_date: str,
        article_count: int = 5,
        previous_searches: list[dict] = None,
    ) -> ESearchRequest | None:
        """
        Generates PubMed search parameters
        """
        user_content = GENERATE_QUERY_USER.format(
            interests=", ".join(interests),
            negative_keywords=negative_keywords,
            date=search_from_date,
            previous_searches=json.dumps(previous_searches, indent=2)
            if previous_searches
            else "None",
        )
        sys_instr = (
            GENERATE_QUERY_SYS + f"\n\nPUBMED API EXAMPLES:\n{ESearchRequest.__doc__}"
        )

        try:
            response = await self.client.chat.send_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_instr},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ESearchRequest",
                        "strict": False,
                        "schema": LLMPubMedQuery.model_json_schema(),
                    },
                },
            )

            llm_result = LLMPubMedQuery.model_validate_json(
                response.choices[0].message.content
            )

            req_obj = ESearchRequest(
                db="pubmed",
                term=llm_result.term,
                retmax=article_count,
                mindate=search_from_date,
                retmode="json",
            )

            return req_obj
        except Exception as e:
            logger.error(f"Error generating PubMed query: {e}")
            return None

    @traceable(name="summarize_article")
    async def summarize_article(self, article_data: dict[str, Any]):
        """Summarizes the given article data."""
        user_content = SUMMARIZE_ARTICLE_USER.format(article_data=article_data)

        try:
            response = await self.client.chat.send_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARIZE_ARTICLE_SYS},
                    {"role": "user", "content": user_content},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error summarizing article: {e}")
            return None

    @traceable(name="format_email")
    async def format_email(self, user_profile: UserProfile, summaries: List[Summary]):
        """
        Formats the email content based on the summary
        """
        try:
            for summary in summaries:
                summary["rating_links_html"] = self._create_rating_links(
                    user_profile.id, summary["pmid"]
                )

            user_content = FORMAT_EMAIL_USER.format(
                first_name=user_profile.first_name,
                user_profile=str(user_profile),
                summaries_json=json.dumps([dict(s) for s in summaries], indent=2),
                unsubscribe_link=self._create_unsubscribe_link(user_profile.id),
            )

            response = await self.client.chat.send_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": FORMAT_EMAIL_SYS},
                    {"role": "user", "content": user_content},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error formatting email for {user_profile.id}: {e}")
            return None

    @traceable(name="extract_article_intent")
    async def extract_article_intent(
        self, article_title: str, article_summary: str
    ) -> str:
        """get core intent of article as a paragraph"""
        user_content = f"Title: {article_title}\n\nSummary/Abstract: {article_summary}"

        try:
            response = await self.client.chat.send_async(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXTRACT_ARTICLE_INTENT},
                    {"role": "user", "content": user_content},
                ],
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error extracting article intent: {e}")
            return ""

    def generate_embedding(self, text: str) -> list[float]:
        """384-dimensional embedding with fastembed"""

        embeddings = list(self.embedding_model.embed([text]))

        if embeddings:
            return embeddings[0].tolist()

        return []

    def _create_rating_links(self, user_id: str, article_id: str) -> str:
        links = [
            f'<a href="{self.feedback_base_url}?user_id={user_id}&pmid={article_id}&rating={i}" '
            f'style="text-decoration: none; margin: 0 5px; font-size: 1.2em; color: #007bff;">{i}</a>'
            for i in range(1, 6)
        ]

        html_string = " ".join(links)

        return f"<b>How relevant was this?</b><br>{html_string}<br><small>(1=Not Relevant, 5=Very Relevant)</small>"

    def _create_unsubscribe_link(self, user_id: str) -> str:
        url = f"{self.unsubscribe_base_url}?user_id={user_id}"
        return f'<a href="{url}">Unsubscribe</a>'
