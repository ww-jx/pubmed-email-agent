import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.pubmed_email_agent.tools.user.client import UserProfile, UserTools

TEST_DB_URL = os.getenv(
    "TEST_DB_URL", "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"
)

TEST_UUID = "11111111-1111-1111-1111-111111111111"


@pytest_asyncio.fixture
async def user_tools():
    """Test UserTools instance and DB cleanup"""
    tools = UserTools(
        db_connection_string=TEST_DB_URL,
        user_table="users",
        feedback_table="article_feedback",
        email_api_key="mock_key",
        from_email="test@example.com",
    )

    engine = tools.engine
    async with AsyncSession(engine) as session:
        await session.execute(
            text(f"""
            INSERT INTO users (id, email, "firstName", "lastName", country, city, gender, conditions, subscribed)
            VALUES ('{TEST_UUID}', 'test@example.com', 'John', 'Doe', 'USA', 'Boston', 'Male', ARRAY['diabetes'], true)
            ON CONFLICT (id) DO NOTHING;
        """)
        )

        await session.execute(
            text(f"""
            INSERT INTO article_feedback (id, user_id, pmid, rating, created_at)
            VALUES (1, '{TEST_UUID}', '12345678', 5, NOW())
            ON CONFLICT (id) DO NOTHING;
        """)
        )
        await session.commit()

    yield tools

    # test isolation
    async with AsyncSession(engine) as session:
        await session.execute(
            text(f"DELETE FROM article_feedback WHERE user_id = '{TEST_UUID}'")
        )
        await session.execute(text(f"DELETE FROM users WHERE id = '{TEST_UUID}'"))
        await session.commit()

    await engine.dispose()


@pytest.mark.asyncio
async def test_get_user_profile_success(user_tools):
    """Real user test"""
    profile = await user_tools.get_user_profile(TEST_UUID)
    assert profile is not None
    assert isinstance(profile, UserProfile)
    assert profile.email == "test@example.com"
    assert "diabetes" in profile.conditions


@pytest.mark.asyncio
async def test_get_user_profile_sql_injection_safety(user_tools):
    """SQL injection test"""
    malicious_id = "11111111-1111-1111-1111-111111111111'; DROP TABLE users; --"
    # Should gracefully catch the DBAPIError inside the tool and return None
    profile = await user_tools.get_user_profile(malicious_id)
    assert profile is None
