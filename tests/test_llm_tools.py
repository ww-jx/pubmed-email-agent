import re
import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qsl, urlparse

import pytest

from src.pubmed_email_agent.tools.llm.client import LLMTools


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
        feedback_function_url="https://test.com/feedback",
        unsubscribe_function_url="https://test.com/unsubscribe",
        link_signing_secret="test-signing-secret",
    )


@pytest.mark.asyncio
async def test_generate_search_query_success(llm_tools):
    mock_json = '{"term": "diabetes AND treatment"}'

    mock_message = MagicMock()
    mock_message.content = mock_json
    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    llm_tools.client.chat.send_async.return_value = mock_response

    with (
        patch(
            "src.pubmed_email_agent.tools.llm.client.LLMPubMedQuery"
        ) as MockLLMPubMedQuery,
        patch(
            "src.pubmed_email_agent.tools.llm.client.ESearchRequest"
        ) as MockESearchRequest,
    ):
        mock_llm_result = MagicMock()
        mock_llm_result.term = "diabetes AND treatment"
        MockLLMPubMedQuery.model_validate_json.return_value = mock_llm_result
        MockLLMPubMedQuery.model_json_schema.return_value = {}

        expected_obj = MagicMock()
        MockESearchRequest.return_value = expected_obj

        result = await llm_tools.generate_search_query(
            ["diabetes"], ["diet"], ("2024/01/01", "2024/06/15"), 5
        )

        assert result == expected_obj
        llm_tools.client.chat.send_async.assert_called_once()

        # mindate alone is ignored by E-utilities; it only filters as a closed
        # range with maxdate and datetype.
        MockESearchRequest.assert_called_once_with(
            db="pubmed",
            term="diabetes AND treatment",
            retmax=5,
            mindate="2024/01/01",
            maxdate="2024/06/15",
            datetype="edat",
            retmode="json",
        )


@pytest.mark.asyncio
async def test_generate_search_query_invalid_json(llm_tools, caplog):
    bad_json = "{bad_json: missing_quotes}"
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=bad_json))]
    llm_tools.client.chat.send_async.return_value = mock_response

    result = await llm_tools.generate_search_query(
        ["diabetes"], [], ("2024/01/01", "2024/06/15"), 5
    )

    assert result is None
    assert "Error generating PubMed query" in caplog.text


def _query(html: str) -> dict[str, str]:
    """The query parameters of the first href in a fragment."""
    href = re.search(r'href="([^"]+)"', html).group(1)
    return dict(parse_qsl(urlparse(href).query))


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


def test_signed_links_reject_tampering(llm_tools):
    """
    The token must cover every parameter, not just the user id.

    An unsigned link let anyone unsubscribe a reader or file ratings in their
    name, so this pins that changing any signed value invalidates the token --
    the same check the edge functions run in
    supabase/functions/_shared/signed-link.ts.
    """
    params = _query(llm_tools._create_rating_links("user-123", "paper-456"))
    token = params.pop("token")

    assert llm_tools._sign(params) == token

    for key, tampered in [
        ("user_id", "someone-else"),
        ("pmid", "999"),
        ("rating", "5"),
        ("exp", str(int(params["exp"]) + 86400)),
    ]:
        assert llm_tools._sign(params | {key: tampered}) != token


def test_links_carry_an_expiry(llm_tools):
    """A link leaked from a forwarded email has to stop working eventually."""
    issued = _query(llm_tools._create_unsubscribe_link("user-123"))

    assert int(issued["exp"]) > time.time()


def test_empty_signing_secret_is_refused(mock_openrouter):
    """An empty HMAC key still produces a token, so it has to fail loudly."""
    with pytest.raises(ValueError, match="link_signing_secret is empty"):
        LLMTools(
            client=mock_openrouter,
            model="test-model",
            feedback_function_url="https://test.com/feedback",
            unsubscribe_function_url="https://test.com/unsubscribe",
            link_signing_secret="",
        )
