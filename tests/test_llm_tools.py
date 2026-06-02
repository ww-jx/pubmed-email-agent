import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.user.client import UserProfile


@pytest.fixture
def mock_openrouter():
    """mock OpenRouter client"""
    client = MagicMock()
    client.chat.send_async = AsyncMock()
    return client


@pytest.fixture
def llm_tools(mock_openrouter):
    return LLMTools(
        client=mock_openrouter,
        model="test-model",
        feedback_base_url="https://test.com/feedback",
        unsubscribe_base_url="https://test.com/unsubscribe",
    )


@pytest.fixture
def sample_user_profile():
    return UserProfile(
        id="test-id",
        email="test@example.com",
        first_name="John",
        last_name="Doe",
        country="USA",
        city="Boston",
        gender="Male",
        conditions=["diabetes"],
        last_email_date=None,
        subscribed=True,
    )


@pytest.mark.asyncio
async def test_generate_search_query_success(llm_tools):
    mock_json = '{"term": "diabetes AND treatment", "retmax": 5}'

    mock_message = MagicMock()
    mock_message.content = mock_json
    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    llm_tools.client.chat.send_async.return_value = mock_response

    with patch(
        "src.pubmed_email_agent.tools.llm.client.ESearchRequest"
    ) as MockESearchRequest:
        expected_obj = MagicMock()
        MockESearchRequest.model_json_schema.return_value = {}
        MockESearchRequest.model_validate_json.return_value = expected_obj

        result = await llm_tools.generate_search_query(
            ["diabetes"], ["diet"], "2024-01-01", 5
        )

        assert result == expected_obj
        llm_tools.client.chat.send_async.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_article_success(llm_tools):
    mock_text = "This is a great summary."
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=mock_text))]
    llm_tools.client.chat.send_async.return_value = mock_response

    result = await llm_tools.summarize_article(
        {"title": "Test Paper", "abstract": "Data"}
    )

    assert result == mock_text


@pytest.mark.asyncio
async def test_format_email_success(llm_tools, sample_user_profile):
    mock_text = "<html><body>Your Email</body></html>"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=mock_text))]
    llm_tools.client.chat.send_async.return_value = mock_response

    summaries = [{"pmid": "123", "title": "Paper 1", "summary": "Abstract summary"}]

    result = await llm_tools.format_email(sample_user_profile, summaries)

    assert result == mock_text
    assert "rating_links_html" in summaries[0]


@pytest.mark.asyncio
async def test_generate_search_query_api_error(llm_tools, caplog):
    llm_tools.client.chat.send_async.side_effect = Exception(
        "OpenRouter Rate Limit Exceeded"
    )

    result = await llm_tools.generate_search_query(["diabetes"], [], "2024-01-01", 5)

    assert result is None
    assert "Error generating PubMed query" in caplog.text
    assert "OpenRouter Rate Limit Exceeded" in caplog.text


@pytest.mark.asyncio
async def test_generate_search_query_invalid_json(llm_tools, caplog):
    bad_json = "{bad_json: missing_quotes}"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=bad_json))]
    llm_tools.client.chat.send_async.return_value = mock_response

    result = await llm_tools.generate_search_query(["diabetes"], [], "2024-01-01", 5)

    assert result is None
    assert "Error generating PubMed query" in caplog.text


@pytest.mark.asyncio
async def test_summarize_article_api_error(llm_tools, caplog):
    llm_tools.client.chat.send_async.side_effect = Exception("API Timeout")

    result = await llm_tools.summarize_article({"title": "Test"})

    assert result is None
    assert "Error summarizing article" in caplog.text


@pytest.mark.asyncio
async def test_format_email_api_error(llm_tools, sample_user_profile, caplog):
    llm_tools.client.chat.send_async.side_effect = Exception("Context Window Exceeded")

    result = await llm_tools.format_email(sample_user_profile, [{"pmid": "123"}])

    assert result is None
    assert "Error formatting email" in caplog.text


def test_create_rating_links(llm_tools):
    html = llm_tools._create_rating_links("user-123", "paper-456")

    assert "https://test.com/feedback" in html
    assert "user_id=user-123" in html
    assert "pmid=paper-456" in html
    assert "rating=1" in html
    assert "rating=5" in html


def test_create_unsubscribe_link(llm_tools):
    html = llm_tools._create_unsubscribe_link("user-123")

    assert "https://test.com/unsubscribe" in html
    assert "user_id=user-123" in html
    assert ">Unsubscribe<" in html
