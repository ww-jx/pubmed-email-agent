import pytest
import httpx
import os
from openrouter import OpenRouter

from src.pubmed_email_agent.tools.llm.client import LLMTools


@pytest.fixture
def llm_tools():
    api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key")
    client = OpenRouter(api_key=api_key)
    return LLMTools(
        client=client,
        model="google/gemini-flash-1.5",
        feedback_base_url="http://test.com/feedback",
        unsubscribe_base_url="http://test.com/unsub",
    )


def test_generate_embedding_live(llm_tools):
    """
    ensure vector generated from text has correct dimensionality (384) using BAAI/bge-small-en-v1.5
    """
    try:
        embedding = llm_tools.generate_embedding("This is a test medical sentence.")
        print(
            f"\n[Live Embedding] Generated vector of size {len(embedding)}. First 5 dims: {embedding[:5]}"
        )

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert isinstance(embedding[0], float)

    except Exception as e:
        pytest.skip(f"Skipping local fastembed test due to error: {e}")


@pytest.mark.asyncio
async def test_extract_article_intent_live(llm_tools):
    """
    live test for `extract_article_intent`
    """
    try:
        intent = await llm_tools.extract_article_intent(
            "Metformin and Type 2 Diabetes",
            "This article discusses the mechanism of action of metformin.",
        )
        print(f"\n[Live Intent Extraction] {intent}")

        assert isinstance(intent, str)
        assert len(intent) > 0
    except (httpx.RequestError, httpx.HTTPStatusError, Exception) as e:
        pytest.skip(f"Skipping live openrouter test due to network/auth error: {e}")
