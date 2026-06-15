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
    """verifies feedback node extracts intent and returns citation graph pmids"""
    profile = UserProfile(
        id="u1",
        email="e@e.com",
        first_name="A",
        last_name="B",
        country="C",
        city="D",
        gender="M",
        conditions=["Cancer"],
        last_email_date=date.today(),
        subscribed=True,
        feedback=[
            UserFeedback(
                id=1, created_at=date.today(), user_id="u1", pmid="123", rating=5
            )
        ],
    )
    state = AgentState(user_profile=profile)

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

    agent.llm_tools.extract_article_intent.assert_called_once_with(
        "Test Title", "Test Abstract"
    )

    assert "111" in result["article_ids"]
    assert result["feedback_intent"] == "intent string"


@pytest.mark.asyncio
async def test_fetch_article_details(agent):
    """verifies _fetch_article_details just fetches metadata for master list"""
    state = AgentState(article_ids=["456"])

    agent.pubmed_tools.fetch.return_value = [{"fetched": "data"}]

    result = await agent._fetch_article_details(state)

    assert len(result["fetched_articles"]) == 1


@pytest.mark.asyncio
async def test_rank_articles(agent):
    """verifies _rank_articles computes cosine similarity and slices top N"""
    profile = UserProfile(
        id="u1",
        email="e@e.com",
        first_name="A",
        last_name="B",
        country="C",
        city="D",
        gender="M",
        conditions=["Cancer"],
        last_email_date=date.today(),
        subscribed=True,
        feedback=[],
    )

    agent.llm_tools.compute_cosine_similarity = MagicMock(side_effect=[0.5, 0.9])

    state = AgentState(
        user_profile=profile,
        feedback_intent="intent string",
        fetched_articles=[
            {
                "MedlineCitation": {
                    "PMID": {"#text": "111"},
                    "Article": {
                        "ArticleTitle": "Bad Title",
                        "Abstract": {"AbstractText": "Bad"},
                    },
                }
            },
            {
                "MedlineCitation": {
                    "PMID": {"#text": "999"},
                    "Article": {
                        "ArticleTitle": "Good Title",
                        "Abstract": {"AbstractText": "Good"},
                    },
                }
            },
        ],
    )

    agent.article_count = 1

    result = await agent._rank_articles(state)

    await asyncio.sleep(0.1)

    agent.llm_tools.generate_embedding.assert_any_call("Cancer intent string")

    assert agent.user_tools.upsert_article_embedding.call_count == 2

    assert result["article_ids"] == ["999"]
    assert len(result["fetched_articles"]) == 1
