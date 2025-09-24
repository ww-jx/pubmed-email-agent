from typing import AsyncIterator

import httpx
import xmltodict
from contextlib import asynccontextmanager

from pubmedclient.sdk import esearch, efetch
from pubmedclient.models import ESearchRequest, EFetchRequest, Db


class PubmedTools:
    def __init__(self, tool_name: str, email: str):
        self.tool_name = tool_name
        self.email = email

        self.db = Db.PUBMED

    async def search(self, params: ESearchRequest) -> list[str]:
        """
        Searches PubMed with the given parameters and returns a list of article ids.
        """

        params.db = self.db.value

        async with self._create_http_client() as client:
            response = await esearch(client, params)
            return response.esearchresult.idlist

    async def fetch(self, ids: list[str]) -> list[dict]:
        """
        Retrieves articles from PubMed given a list of article ids.

        Data is returned in XML format and parsed into a list of dictionaries.
        """
        params = EFetchRequest(
            db=Db.PUBMED,
            id=",".join(ids),
            rettype="xml",  # abstract, medline, xml
            retmode="xml",  # text, xml
        )

        async with self._create_http_client() as client:
            response = await efetch(client, params)

            parsed_response = xmltodict.parse(response)

            return parsed_response.get("PubmedArticleSet", {}).get("PubmedArticle", [])

    @asynccontextmanager
    async def _create_http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if not self.tool_name or not self.email:
            raise ValueError("Tool name and email must be provided")

        headers = {
            "tool": self.tool_name,
            "email": self.email,
        }

        async with httpx.AsyncClient(headers=headers) as client:
            yield client
