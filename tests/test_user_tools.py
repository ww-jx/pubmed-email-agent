import os
import pytest
import pytest_asyncio
from unittest.mock import patch
from datetime import date
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.pubmed_email_agent.tools.user.client import UserTools, UserProfile

TEST_DB_URL = os.getenv(
    "TEST_DB_URL", "postgresql+asyncpg://postgres:postgres@127.0.0.1:54322/postgres"
)

TEST_UUID = "11111111-1111-1111-1111-111111111111"
FAKE_UUID = "99999999-9999-9999-9999-999999999999"


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
async def test_get_all_user_ids(user_tools):
    """Fetch all users"""
    user_ids = await user_tools.get_all_user_ids()
    assert isinstance(user_ids, list)
    assert len(user_ids) >= 1
    assert TEST_UUID in user_ids


@pytest.mark.asyncio
async def test_get_user_profile_success(user_tools):
    """Real user test"""
    profile = await user_tools.get_user_profile(TEST_UUID)
    assert profile is not None
    assert isinstance(profile, UserProfile)
    assert profile.email == "test@example.com"
    assert "diabetes" in profile.conditions


@pytest.mark.asyncio
async def test_get_user_profile_not_found(user_tools):
    """Fake user test"""
    profile = await user_tools.get_user_profile(FAKE_UUID)
    assert profile is None


@pytest.mark.asyncio
async def test_update_last_email_date(user_tools):
    """Timestamp update test"""
    success = await user_tools.update_last_email_date(TEST_UUID)
    assert success is True
    profile = await user_tools.get_user_profile(TEST_UUID)
    assert profile.last_email_date == date.today()


@pytest.mark.asyncio
async def test_get_user_feedback(user_tools):
    """Get feedback test"""
    feedback = await user_tools.get_user_feedback(TEST_UUID)
    assert len(feedback) == 1
    assert feedback[0].pmid == "12345678"
    assert feedback[0].rating == 5


@pytest.mark.asyncio
async def test_get_user_feedback_empty(user_tools):
    """No feedback test"""
    feedback = await user_tools.get_user_feedback(FAKE_UUID)
    assert isinstance(feedback, list)
    assert len(feedback) == 0


@pytest.mark.asyncio
async def test_get_user_profile_sql_injection_safety(user_tools):
    """SQL injection test"""
    malicious_id = "11111111-1111-1111-1111-111111111111'; DROP TABLE users; --"
    # Should gracefully catch the DBAPIError inside the tool and return None
    profile = await user_tools.get_user_profile(malicious_id)
    assert profile is None


# DB DOWN TESTS
@pytest.mark.asyncio
async def test_get_all_user_ids_db_failure(user_tools, caplog):
    with patch("src.pubmed_email_agent.tools.user.client.AsyncSession") as mock_session:
        mock_session.return_value.__aenter__.side_effect = Exception(
            "Simulated DB Outage"
        )

        result = await user_tools.get_all_user_ids()
        assert result == []
        assert "Simulated DB Outage" in caplog.text


@pytest.mark.asyncio
async def test_update_last_email_date_db_failure(user_tools, caplog):
    with patch("src.pubmed_email_agent.tools.user.client.AsyncSession") as mock_session:
        mock_session.return_value.__aenter__.side_effect = Exception(
            "Simulated DB Outage"
        )

        result = await user_tools.update_last_email_date(TEST_UUID)
        assert result is False
        assert "Simulated DB Outage" in caplog.text


@pytest.mark.asyncio
async def test_get_user_feedback_db_failure(user_tools, caplog):
    with patch("src.pubmed_email_agent.tools.user.client.AsyncSession") as mock_session:
        mock_session.return_value.__aenter__.side_effect = Exception(
            "Simulated DB Outage"
        )

        result = await user_tools.get_user_feedback(TEST_UUID)
        assert result == []
        assert "Simulated DB Outage" in caplog.text


@pytest.mark.asyncio
async def test_get_user_profile_db_failure(user_tools, caplog):
    with patch("src.pubmed_email_agent.tools.user.client.AsyncSession") as mock_session:
        mock_session.return_value.__aenter__.side_effect = Exception(
            "Simulated DB Outage"
        )

        result = await user_tools.get_user_profile(TEST_UUID)
        assert result is None
        assert "DB error or invalid format" in caplog.text


def test_send_email_failure(user_tools, caplog):
    """Email API failure test"""
    with patch.object(
        user_tools.email_client, "send", side_effect=Exception("API Down")
    ):
        result = user_tools.send_email(
            to_email="test@example.com", subject="Test", body="Test"
        )
        assert result is False
        assert "Error sending email" in caplog.text
