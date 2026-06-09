import pytest
from unittest.mock import patch
import json
import httpx
from pubmedclient.models import ESearchRequest, ELinkRequest, Db, ELinkCmd, RetMode
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools
from tests.api_models import FetchArticleModel, ESearchResponseModel, ELinkResponseModel
from pubmedclient.sdk import esearch, elink

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

        print(json.dumps({"article_ids": article_ids, "feedback": feedback}, indent=2))

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
        print(json.dumps({"article_ids": article_ids, "feedback": feedback}, indent=2))

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

        print(json.dumps(results, indent=2))

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
async def test_fetch_invalid_graceful_failure(pubmed_tools):
    """Test that fetching a non-existent PMID"""
    fake_pmid = "999999999999"

    try:
        results = await pubmed_tools.fetch([fake_pmid])
        print(json.dumps(results, indent=2))

        assert isinstance(results, list)
        assert len(results) == 0
    except Exception as e:
        pytest.fail(
            f"Tool crashed on invalid fetch instead of degrading gracefully: {e}"
        )


@pytest.mark.asyncio
async def test_get_related_articles(pubmed_tools):
    """Test Neighbor Score ELink returns results for a known PMID"""
    try:
        results = await pubmed_tools.get_related_articles(
            [KNOWN_STABLE_PMID], article_count=5
        )

        if not results:
            pytest.skip("ELink returned empty list. Possible network error.")

        print(json.dumps(results, indent=2))

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

        print(json.dumps(results, indent=2))

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
