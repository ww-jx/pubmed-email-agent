from typing import TypedDict, List
from pubmedclient.models import ESearchRequest

from src.pubmed_email_agent.tools.user.client import UserProfile


class Summary(TypedDict):
    pmid: str
    title: str
    link: str
    summary: str


class AgentState(TypedDict):
    user_id: str
    user_profile: UserProfile
    search_from_date: str
    search_request: ESearchRequest
    article_ids: List[str]
    fetched_articles: List[dict]
    summaries: List[Summary]
    email_content: str
    retries: int
