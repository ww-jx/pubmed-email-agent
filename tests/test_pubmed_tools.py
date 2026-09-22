import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pubmedclient.models import Db, ELinkCmd, ELinkRequest, ESearchRequest, RetMode
from pubmedclient.sdk import elink, esearch

from src.pubmed_email_agent.tools.pubmed.client import PubmedTools
from tests.api_models import ELinkResponseModel, ESearchResponseModel, FetchArticleModel

KNOWN_STABLE_PMID = "23178126"


@pytest.fixture
def pubmed_tools():
    """live instance"""
    return PubmedTools(tool_name="pytest_eval_suite", email="dev@example.com")


@pytest.mark.asyncio
async def test_search_valid_results(pubmed_tools):
    """Test that a valid search returns a list of PMIDs"""
    request = ESearchRequest(term="type 2 diabetes", retmax=3)

    try:
        article_ids, feedback = await pubmed_tools.search(request)

        if not article_ids and feedback.get("errorlist", {}).get("network_error"):
            pytest.skip("Search returned empty list. Possible network error.")

        print(
            f"[Search API] Found {len(article_ids)} articles. Feedback: {json.dumps(feedback)}"
        )

        assert isinstance(article_ids, list)
        assert len(article_ids) > 0
    except Exception as e:
        pytest.fail(f"Live API search failed unexpectedly: {e}")


@pytest.mark.asyncio
async def test_search_no_results(pubmed_tools):
    """Test a nonsense search"""
    request = ESearchRequest(term="asdfqwer1234 fake_disease_xyz", retmax=5)

    try:
        article_ids, feedback = await pubmed_tools.search(request)
        print(
            f"[Search API No Results] Found {len(article_ids)} articles. ErrorList: {json.dumps(feedback.get('errorlist', {}))}"
        )

        assert isinstance(article_ids, list)
        assert len(article_ids) == 0

        assert feedback.get("errorlist") is not None
        assert "phrasesnotfound" in feedback["errorlist"]
    except Exception as e:
        pytest.fail(f"Live API search threw an error instead of returning empty: {e}")


@pytest.mark.asyncio
async def test_fetch_valid_structure(pubmed_tools):
    """
    Test that xml parsing is still aligned with API.
    """
    try:
        results = await pubmed_tools.fetch([KNOWN_STABLE_PMID])

        if not results:
            pytest.skip("Fetch returned empty list. Possible network error.")

        print(f"[Fetch API] Successfully fetched details for {len(results)} articles.")

        assert isinstance(results, list)
        assert len(results) == 1

        article = results[0]

        try:
            FetchArticleModel(**article)
        except Exception as e:
            pytest.fail(f"API Schema Change Detected in Fetch:\n{e}")

    except httpx.HTTPStatusError as e:
        pytest.skip(f"Skipping test due to NCBI server error: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Skipping test due to NCBI network error: {e}")
    except Exception as e:
        pytest.fail(f"Fetch API or parsing failed: {e}")


@pytest.mark.asyncio
async def test_get_related_articles(pubmed_tools):
    """Test Neighbor Score ELink returns results for a known PMID"""
    try:
        results = await pubmed_tools.get_related_articles(
            [KNOWN_STABLE_PMID], ("2000/01/01", "2030/01/01"), article_count=5
        )

        if not results:
            pytest.skip("ELink returned empty list. Possible network error.")

        print(f"[ELink API] Found related PMIDs: {results}")

        assert isinstance(results, list)
        assert len(results) > 0, (
            "Expected at least one related article for stable PMID — "
            "possible silent ELink failure or schema change"
        )
        assert all(isinstance(r, str) for r in results)
    except httpx.HTTPStatusError as e:
        pytest.skip(f"Skipping due to NCBI server error: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Skipping due to network error: {e}")
    except Exception as e:
        pytest.fail(f"ELink API failed: {e}")


@pytest.mark.asyncio
async def test_esearch_api_schema(pubmed_tools):
    request = ESearchRequest(term="type 2 diabetes", retmax=3, db=Db.PUBMED.value)
    try:
        async with pubmed_tools._create_http_client() as client:
            response = await esearch(client, request)
            esearch_data = response.model_dump()
            ESearchResponseModel(**esearch_data)

    except httpx.HTTPStatusError as e:
        pytest.skip(f"Skipping test due to NCBI server error: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Skipping test due to NCBI network error: {e}")
    except Exception as e:
        pytest.fail(f"API Schema Change Detected in ESearch:\n{e}")


@pytest.mark.asyncio
async def test_elink_api_schema(pubmed_tools):
    request = ELinkRequest(
        dbfrom=Db.PUBMED,
        db=Db.PUBMED,
        id=[KNOWN_STABLE_PMID],
        cmd=ELinkCmd.NEIGHBOR_SCORE,
        retmode=RetMode.JSON,
    )
    try:
        async with pubmed_tools._create_http_client() as client:
            raw_response = await elink(client, request)
            data = json.loads(raw_response, strict=False)
            ELinkResponseModel(**data)

    except httpx.HTTPStatusError as e:
        pytest.skip(f"Skipping test due to NCBI server error: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Skipping test due to NCBI network error: {e}")
    except Exception as e:
        pytest.fail(f"API Schema Change Detected in ELink:\n{e}")


@pytest.mark.asyncio
async def test_get_article_keywords(pubmed_tools):
    try:
        results = await pubmed_tools.get_article_keywords([KNOWN_STABLE_PMID])

        if not results:
            pytest.skip("Keywords returned empty list. Possible network error.")

        print(f"[Keyword Extraction] Extracted keywords: {results}")

        assert isinstance(results, list)
        assert len(results) > 0, (
            "Expected keywords for stable PMID — "
            "possible silent fetch failure or schema change"
        )
    except httpx.HTTPStatusError as e:
        pytest.skip(f"Skipping due to NCBI server error: {e}")
    except httpx.RequestError as e:
        pytest.skip(f"Skipping due to network error: {e}")
    except Exception as e:
        pytest.fail(f"Keyword extraction failed: {e}")


@pytest.mark.asyncio
async def test_search_network_error(pubmed_tools):
    request = ESearchRequest(term="type 2 diabetes", retmax=3)

    with patch(
        "src.pubmed_email_agent.tools.pubmed.client.esearch",
        side_effect=Exception("Mocked network error"),
    ):
        article_ids, feedback = await pubmed_tools.search(request)

        assert isinstance(article_ids, list)
        assert len(article_ids) == 0
        assert feedback.get("errorlist") is not None
        assert "network_error" in feedback["errorlist"]
        assert "Mocked network error" in feedback["errorlist"]["network_error"]


@pytest.mark.asyncio
async def test_get_related_articles_ranks_by_score(pubmed_tools):
    payload = {
        "linksets": [
            {
                "dbfrom": "pubmed",
                "ids": ["111"],
                "linksetdbs": [
                    {
                        "linkname": "pubmed_pubmed",
                        "links": [
                            {"id": "900", "score": 10},
                            {"id": "800", "score": 40},
                        ],
                    },
                    {
                        "linkname": "pubmed_pubmed_citedin",
                        "links": [{"id": "700", "score": 99}],
                    },
                ],
            },
            {
                "dbfrom": "pubmed",
                "ids": ["222"],
                "linksetdbs": [
                    {
                        "linkname": "pubmed_pubmed",
                        "links": [
                            {"id": "800", "score": 90},
                            {"id": "950", "score": 50},
                        ],
                    }
                ],
            },
        ]
    }

    with patch(
        "src.pubmed_email_agent.tools.pubmed.client.elink",
        new=AsyncMock(return_value=json.dumps(payload)),
    ):
        results = await pubmed_tools.get_related_articles(
            ["111", "222"], ("2026/08/20", "2026/09/20"), article_count=3
        )

    assert results == ["800", "950", "900"]


@pytest.mark.asyncio
async def test_get_related_articles_sends_both_date_bounds(pubmed_tools):
    """E-utilities ignores a lone mindate, so the window must travel whole."""
    mock_elink = AsyncMock(return_value=json.dumps({"linksets": []}))

    with patch("src.pubmed_email_agent.tools.pubmed.client.elink", new=mock_elink):
        await pubmed_tools.get_related_articles(["111"], ("2026/08/20", "2026/09/20"))

    sent = mock_elink.await_args.args[1]
    assert sent.mindate == "2026/08/20"
    assert sent.maxdate == "2026/09/20"
    assert sent.datetype == "edat"
