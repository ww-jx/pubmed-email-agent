from typing import Any, List

import json
from pubmedclient.models import ESearchRequest
from openrouter import OpenRouter
from langsmith import traceable

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

    @traceable(name="generate_search_query")
    async def generate_search_query(
        self,
        interests: list[str],
        negative_keywords: list[str],
        search_from_date: str,
        article_count: int = 5,
    ) -> ESearchRequest:
        """
        Generates PubMed search parameters
        """
        user_content = GENERATE_QUERY_USER.format(
            interests=", ".join(interests),
            negative_keywords=negative_keywords,
            date=search_from_date,
            article_count=article_count,
        )

        response = await self.client.chat.send_async(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATE_QUERY_SYS},
                {"role": "user", "content": user_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ESearchRequest",
                    "strict": True,
                    "schema": ESearchRequest.model_json_schema(),
                },
            },
        )

        return ESearchRequest.model_validate_json(response.choices[0].message.content)

    @traceable(name="summarize_article")
    async def summarize_article(self, article_data: dict[str, Any]) -> str:
        """Summarizes the given article data."""
        user_content = SUMMARIZE_ARTICLE_USER.format(article_data=article_data)

        response = await self.client.chat.send_async(
            model=self.model,
            messages=[
                {"role": "system", "content": SUMMARIZE_ARTICLE_SYS},
                {"role": "user", "content": user_content},
            ],
        )

        return response.choices[0].message.content

    @traceable(name="format_email")
    async def format_email(
        self, user_profile: UserProfile, summaries: List[Summary]
    ) -> str:
        """
        Formats the email content based on the summary
        """

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
