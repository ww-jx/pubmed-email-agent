import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import date

from src.pubmed_email_agent.agent.graph import Agent
from src.pubmed_email_agent.agent.state import AgentState
from src.pubmed_email_agent.tools.user.client import UserProfile, UserFeedback


@pytest.fixture
def mock_user_tools():
    tools = MagicMock()
    tools.get_user_profile = AsyncMock()
    tools.get_user_feedback = AsyncMock()
    tools.search_similar_pmids = AsyncMock(return_value=["999"])
    tools.upsert_article_embedding = AsyncMock()
    return tools


@pytest.fixture
def mock_llm_tools():
    tools = MagicMock()
    tools.extract_article_intent = AsyncMock(return_value="intent string")
    tools.generate_embedding = MagicMock(return_value=[0.1] * 384)
    return tools


@pytest.fixture
def mock_pubmed_tools():
    tools = MagicMock()
    tools.fetch = AsyncMock()
    tools.get_related_articles = AsyncMock(return_value=["111"])
    tools.get_article_keywords = AsyncMock(return_value=[])

    def fake_parse_article(article_dict):
        pmid = article_dict.get("MedlineCitation", {}).get("PMID", {}).get("#text", "")
        title = (
            article_dict.get("MedlineCitation", {})
            .get("Article", {})
            .get("ArticleTitle", "")
        )
        abstract = (
            article_dict.get("MedlineCitation", {})
            .get("Article", {})
            .get("Abstract", {})
            .get("AbstractText", "")
        )
        return {"pmid": pmid, "title": title, "abstract": abstract}

    tools.parse_article = MagicMock(side_effect=fake_parse_article)
    return tools


@pytest.fixture
def agent(mock_user_tools, mock_llm_tools, mock_pubmed_tools):
    return Agent(
        user_tools=mock_user_tools,
        llm_tools=mock_llm_tools,
        pubmed_tools=mock_pubmed_tools,
    )


@pytest.mark.asyncio
async def test_process_feedback_with_hybrid_rag(agent):
    """verifies hybrid rag fetches highly-rated articles, embeds intent, and combines vector-search and citation-graph pmids"""
    # mock user with positive feedback (rating: 5)
    profile = UserProfile(
        id="u1",
        email="e@e.com",
        first_name="A",
        last_name="B",
        country="C",
        city="D",
        gender="M",
        conditions=[],
        last_email_date=date.today(),
        subscribed=True,
        feedback=[
            UserFeedback(
                id=1, created_at=date.today(), user_id="u1", pmid="123", rating=5
            )
        ],
    )
    state = AgentState(user_profile=profile)

    # return mock title and abstrac
    agent.pubmed_tools.fetch.return_value = [
        {
            "MedlineCitation": {
                "PMID": {"#text": "123"},
                "Article": {
                    "ArticleTitle": "Test Title",
                    "Abstract": {"AbstractText": "Test Abstract"},
                },
            }
        }
    ]

    result = await agent._process_feedback(state)
    print(
        f"\n[Hybrid RAG Results] Combined {len(result['article_ids'])} PMIDs: {result['article_ids']}"
    )

    agent.llm_tools.extract_article_intent.assert_called_once_with(
        "Test Title", "Test Abstract"
    )
    agent.llm_tools.generate_embedding.assert_called_once_with("intent string")
    agent.user_tools.search_similar_pmids.assert_called_once()

    assert "111" in result["article_ids"]
    assert "999" in result["article_ids"]


@pytest.mark.asyncio
async def test_fetch_article_details_with_embedding(agent):
    """verifies _fetch_article_details generates embedding and stores in db"""
    state = AgentState(article_ids=["456"])

    agent.pubmed_tools.fetch.return_value = [
        {
            "MedlineCitation": {
                "PMID": {"#text": "456"},
                "Article": {
                    "ArticleTitle": "Fetch Title",
                    "Abstract": {"AbstractText": "Fetch Abstract"},
                },
            }
        }
    ]

    result = await agent._fetch_article_details(state)
    print(
        f"\n[Fetch & Embed] Fetched {len(result['fetched_articles'])} articles and dispatched embedding tasks."
    )

    await asyncio.sleep(0.1)

    agent.llm_tools.generate_embedding.assert_called_once_with(
        "Fetch Title Fetch Abstract"
    )

    agent.user_tools.upsert_article_embedding.assert_called_once_with(
        "456", "Fetch Title", "Fetch Abstract", [0.1] * 384
    )

    assert len(result["fetched_articles"]) == 1
