from typing import TypedDict

from pubmedclient.models import ESearchRequest

from src.pubmed_email_agent.tools.user.client import UserProfile


class Summary(TypedDict):
    pmid: str
    title: str
    link: str
    summary: str
    rating_links_html: str


class AgentState(TypedDict):
    user_id: str
    user_profile: UserProfile
    search_from_date: str
    search_to_date: str
    search_request: ESearchRequest
    article_ids: list[str]
    negative_keywords: list[str]
    fetched_articles: list[dict]
    summaries: list[Summary]
    email_content: str
    retries: int
    previous_searches: list[dict]
    feedback_intent: str
