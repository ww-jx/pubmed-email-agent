from typing import List, Optional
from dataclasses import dataclass

import markdown2
from datetime import date
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession


@dataclass
class UserProfile:
    id: str
    email: str
    first_name: str
    last_name: str
    country: str
    city: str
    gender: str
    conditions: List[str]
    last_email_date: date | None


class UserTools:
    def __init__(
        self,
        db_connection_string: str,
        user_table: str,
        email_api_key: str,
        from_email: str,
    ):
        self.engine = create_async_engine(db_connection_string)
        self.user_table = user_table

        self.email_client = SendGridAPIClient(email_api_key)
        self.from_email = from_email

    async def get_all_user_ids(self) -> List[str]:
        """
        Retrieves a list of all user IDs.
        """
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                text(f"SELECT id FROM {self.user_table} ORDER BY id")
            )

            user_ids = [str(user_id) for user_id in result.scalars().all()]

            print(f"Found {len(user_ids)} users.")
            return user_ids

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieves the profile of a user by their ID and returns a UserProfile object.
        """

        query = text(f"""
            SELECT 
                id, 
                email, 
                "firstName" AS first_name, 
                "lastName" AS last_name, 
                country, 
                city, 
                gender, 
                conditions,
                last_email_date
            FROM {self.user_table} 
            WHERE id = :user_id
        """)

        async with AsyncSession(self.engine) as session:
            result = await session.execute(query, {"user_id": user_id})
            profile_data = result.mappings().first()

        if profile_data:
            return UserProfile(**profile_data)

        return None

    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """
        Sends an email to the specified recipient.
        """

        html_content = markdown2.markdown(body)

        message = Mail(
            from_email=self.from_email,
            to_emails=to_email,
            subject=subject,
            html_content=html_content,
        )

        try:
            self.email_client.send(message)
            return True
        except Exception as e:
            print(f"Error sending email to {to_email}: {e}")
            return False

    async def update_last_email_date(self, user_id: str) -> bool:
        query = text(f"""
            UPDATE {self.user_table}
            SET last_email_date = NOW()
            WHERE id = :user_id
        """)

        try:
            async with AsyncSession(self.engine) as session:
                await session.execute(query, {"user_id": user_id})
                await session.commit()
            return True
        except Exception as e:
            print(f"Error updating last_email_date for user {user_id}: {e}")
            return False
