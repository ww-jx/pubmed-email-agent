import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import date
from pubmedclient.models import ESearchRequest

from src.pubmed_email_agent.agent.graph import Agent
from src.pubmed_email_agent.agent.state import AgentState
from src.pubmed_email_agent.tools.user.client import UserProfile


@pytest.fixture
def mock_user_tools():
    tools = MagicMock()
    tools.get_user_profile = AsyncMock()
    tools.get_user_feedback = AsyncMock()
    tools.update_last_email_date = AsyncMock()
    tools.send_email = MagicMock(return_value=True)
    return tools


@pytest.fixture
def mock_llm_tools():
    tools = MagicMock()
    tools.generate_search_query = AsyncMock()
    tools.summarize_article = AsyncMock()
    tools.format_email = AsyncMock()
    return tools


@pytest.fixture
def mock_pubmed_tools():
    tools = MagicMock()
    tools.search = AsyncMock()
    tools.fetch = AsyncMock()
    tools.get_related_articles = AsyncMock()
    tools.get_article_keywords = AsyncMock()
    return tools


@pytest.fixture
def agent(mock_user_tools, mock_llm_tools, mock_pubmed_tools):
    return Agent(
        user_tools=mock_user_tools,
        llm_tools=mock_llm_tools,
        pubmed_tools=mock_pubmed_tools,
        article_count=5,
        max_retries=3,
    )


@pytest.fixture
def sample_user_profile():
    return UserProfile(
        id="user-123",
        email="test@example.com",
        first_name="Jane",
        last_name="Doe",
        country="UK",
        city="London",
        gender="Female",
        conditions=["Diabetes"],
        last_email_date=date(2025, 1, 1),
        subscribed=True,
        feedback=[],
    )


# routing & conditional tests
def test_subscription_check_continue(agent, sample_user_profile):
    state = AgentState(user_profile=sample_user_profile)
    result = agent._subscription_check(state)
    assert result == "continue"


def test_subscription_check_end(agent, sample_user_profile):
    """ignore unsubscribed users"""
    sample_user_profile.subscribed = False
    state = AgentState(user_profile=sample_user_profile)
    result = agent._subscription_check(state)
    assert result == "end"


def test_article_check_proceed(agent):
    """routing when enough articles found"""
    state = AgentState(article_ids=["1", "2", "3", "4", "5"], retries=0)
    assert agent._article_check(state) == "proceed"


def test_article_check_retry(agent):
    """retry when short on articles and have retries left"""
    state = AgentState(article_ids=["1", "2"], retries=1)
    assert agent._article_check(state) == "retry"


def test_article_check_end_empty(agent):
    """terminate when hit max retries and found nothing"""
    state = AgentState(article_ids=[], retries=3)
    assert agent._article_check(state) == "end"


def test_article_check_proceed_partial(agent):
    """proceed when max retries is hit but some articles were found"""
    state = AgentState(article_ids=["1", "2"], retries=3)
    assert agent._article_check(state) == "proceed"


@pytest.mark.asyncio
async def test_full_agent_run_success(agent, sample_user_profile):
    agent.user_tools.get_user_profile.return_value = sample_user_profile
    agent.user_tools.get_user_feedback.return_value = []

    agent.pubmed_tools.get_related_articles.return_value = []
    agent.pubmed_tools.get_article_keywords.return_value = []

    mock_esearch = ESearchRequest(term="Diabetes", db="pubmed")
    agent.llm_tools.generate_search_query.return_value = mock_esearch

    agent.pubmed_tools.search.return_value = (
        ["101", "102", "103", "104", "105"],
        {"count": "5"},
    )
    agent.pubmed_tools.fetch.return_value = [
        {
            "MedlineCitation": {
                "PMID": {"#text": "101"},
                "Article": {"ArticleTitle": "Test 1"},
            }
        }
    ]

    agent.llm_tools.summarize_article.return_value = "A concise summary."
    agent.llm_tools.format_email.return_value = "<html>Email Content</html>"
    agent.user_tools.update_last_email_date.return_value = True

    initial_state = AgentState(user_id="user-123")

    final_state = await agent.run(initial_state)

    print("\n--- [MOCK AGENT RUN] ---")
    print(f"[User] {final_state['user_id']} ({final_state['user_profile'].first_name})")
    print(f"[Generated Query] {mock_esearch.term}")
    print(
        f"[Articles Fetched] {len(final_state['article_ids'])} items: {final_state['article_ids']}"
    )
    print(f"[Summaries Gen] {len(final_state['summaries'])} generated")
    print(f"[Final Email Snippet] {final_state['email_content'][:50]}...")
    print("-" * 35)

    assert final_state["user_profile"] == sample_user_profile
    assert len(final_state["article_ids"]) == 5
    assert len(final_state["summaries"]) == 1
    assert final_state["email_content"] == "<html>Email Content</html>"

    # ensure send_email node triggered side effect
    agent.user_tools.send_email.assert_called_once_with(
        "test@example.com", "Jane's Weekly PubMed Digest", "<html>Email Content</html>"
    )


@pytest.mark.asyncio
async def test_agent_fails_gracefully_on_missing_user(agent):
    agent.user_tools.get_user_profile.return_value = None

    initial_state = AgentState(user_id="ghost-user-999")

    with pytest.raises(
        ValueError, match="No profile found for user ID: ghost-user-999"
    ):
        await agent.run(initial_state)


@pytest.mark.asyncio
async def test_agent_retry_loop_behavior(agent, sample_user_profile):
    agent.user_tools.get_user_profile.return_value = sample_user_profile
    agent.user_tools.get_user_feedback.return_value = []
    agent.pubmed_tools.get_related_articles.return_value = []
    agent.pubmed_tools.get_article_keywords.return_value = []
    agent.llm_tools.generate_search_query.return_value = ESearchRequest(
        term="bs", db="pubmed"
    )

    agent.pubmed_tools.search.side_effect = [
        (["101", "102"], {"count": "2"}),
        (["103", "104", "105"], {"count": "3"}),
    ]

    agent.pubmed_tools.fetch.return_value = []
    agent.llm_tools.format_email.return_value = "Done"

    initial_state = AgentState(user_id="user-123")
    final_state = await agent.run(initial_state)

    print("\n--- [AGENT RETRY LOOP EVAL] ---")
    print(f"[Retries] {final_state['retries']}")
    print(f"[Articles Fetched] {len(final_state['article_ids'])}")
    print("-" * 31)

    assert final_state["retries"] == 1
    assert len(final_state["article_ids"]) == 5
    assert agent.pubmed_tools.search.call_count == 2
