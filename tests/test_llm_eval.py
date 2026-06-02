import os
import re
import pytest
import httpx
from dotenv import load_dotenv
from openrouter import OpenRouter
from pubmedclient.models import ESearchRequest
from src.pubmed_email_agent.tools.llm.client import LLMTools

load_dotenv()

# synthetic test cases
EVAL_TEST_CASES = [
    {
        "id": "explicit_negation_handling",
        "interests": ["Type 2 Diabetes", "GLP-1 Weight Loss"],
        "negative_keywords": ["mice", "rats"],
        "must_include": ["Diabetes", "GLP-1"],
        "must_negate": ["mice", "rats"],
    },
    {
        "id": "medical_acronym_expansion",
        "interests": ["fMRI Neurofeedback", "tACS"],
        "negative_keywords": ["review"],
        "must_include": ["fMRI", "tACS"],
        "must_negate": ["review"],
    },
]


@pytest.mark.parametrize(
    "model_name",
    [
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash-lite",
    ],
)
@pytest.mark.parametrize("case", EVAL_TEST_CASES, ids=lambda c: c["id"])
@pytest.mark.asyncio
async def test_pubmed_query_generation_eval(model_name, case):
    """
    test query-generation with different models and validated against pubmed
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip(
            "Skipping live evaluation: OPENROUTER_API_KEY missing from environment."
        )

    async with OpenRouter(api_key=api_key) as client:
        llm_tools = LLMTools(
            client=client,
            model=model_name,
            feedback_base_url="http://localhost/feedback",
            unsubscribe_base_url="http://localhost/unsubscribe",
        )

        request_obj = await llm_tools.generate_search_query(
            interests=case["interests"],
            negative_keywords=case["negative_keywords"],
            search_from_date="2025/01/01",
            article_count=5,
        )

        assert request_obj is not None, (
            f"Model {model_name} failed safely but returned None."
        )
        assert isinstance(request_obj, ESearchRequest)
        assert request_obj.retmax == 5

        term_upper = request_obj.term.upper().replace('"', "")

        # behavioral assertions
        for inc in case["must_include"]:
            assert inc.upper() in term_upper, (
                f"Model {model_name} failed to include critical focus term '{inc}': {request_obj.term}"
            )

        for exc in case["must_negate"]:
            pattern = rf"NOT\s*\(?[^)]*\b{re.escape(exc.upper())}\b"
            has_valid_negation = bool(re.search(pattern, term_upper))

            assert has_valid_negation, (
                f"Model {model_name} failed to format negation constraints for '{exc}'. Query: {request_obj.term}"
            )

        pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": request_obj.term,
            "retmode": "json",
            "retmax": 1,
        }

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(pubmed_url, params=params)

        assert response.status_code == 200, (
            "PubMed system interface rejected connection request."
        )

        pubmed_data = response.json()
        assert "esearchresult" in pubmed_data, (
            "PubMed payload changed root metadata wrapper."
        )

        result_details = pubmed_data["esearchresult"]

        assert "errorlist" not in result_details, (
            f"Model {model_name} generated invalid syntax rejected by PubMed: {request_obj.term}"
        )

        count = int(result_details.get("count", 0))
        assert count > 0, (
            f"Model {model_name} generated a valid query, but it returned 0 results. Query: {request_obj.term}"
        )
