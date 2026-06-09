from typing import AsyncIterator, List

import json
import httpx
import random
import xmltodict
from contextlib import asynccontextmanager

from pubmedclient.sdk import esearch, efetch, elink
from pubmedclient.models import (
    ESearchRequest,
    EFetchRequest,
    Db,
    ELinkRequest,
    ELinkCmd,
    RetMode,
)

from src.pubmed_email_agent.logger import get_logger

logger = get_logger(__name__)


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

            pubmed_article_set = parsed_response.get("PubmedArticleSet") or {}
            article_data = pubmed_article_set.get("PubmedArticle", [])

            if not article_data:
                return []

            if isinstance(article_data, dict):
                return [article_data]

            if isinstance(article_data, list):
                return article_data

            return []

    async def get_related_articles(
        self, pmids: List[str], article_count: int = 2
    ) -> List[str]:
        """
        Given a list of PMIDs, retrieves the single most related article PMID for each.

        Returns a dictionary mapping each input PMID to its single most related article PMID
        (or None if none found).
        """
        if not pmids:
            return []

        params = ELinkRequest(
            dbfrom=Db.PUBMED,
            db=Db.PUBMED,
            id=pmids,
            cmd=ELinkCmd.NEIGHBOR_SCORE,
            retmode=RetMode.JSON,
        )

        most_related = []

        async with self._create_http_client() as client:
            try:
                response = await elink(client, params)
                data = json.loads(response)
            except Exception as e:
                logger.error(f"Error fetching related articles: {e}")
                return []

        try:
            linksets = data.get("linksets", [])

            # a lookup map: input PMID -> its linkset object
            pmid_to_linkset = {
                ls.get("ids", [None])[0]: ls
                for ls in linksets
                if ls.get("ids")
                and isinstance(ls.get("ids"), list)
                and len(ls["ids"]) > 0
            }

            for pmid in pmids:
                linkset = pmid_to_linkset.get(pmid)
                if not linkset:
                    continue

                linksetdbs = linkset.get("linksetdbs", [])
                pubmed_pubmed_links = []

                for linksetdb in linksetdbs:
                    if linksetdb.get("linkname") == "pubmed_pubmed":
                        pubmed_pubmed_links = linksetdb.get("links", [])
                        break

                if pubmed_pubmed_links:
                    # sort links by score (desc)
                    def get_score(link):
                        try:
                            return float(link.get("score", -1.0))
                        except (ValueError, TypeError):
                            return -1.0

                    sorted_links = sorted(
                        pubmed_pubmed_links, key=get_score, reverse=True
                    )

                    # get top-scoring id
                    if sorted_links:
                        top_link = sorted_links[0]
                        most_related.append(top_link.get("id"))

        except Exception as e:
            logger.error(f"Error parsing related articles data: {e}")
            raise e

        sample_num = min(article_count, len(most_related))
        random_sample = random.sample(most_related, sample_num)
        return random_sample

    async def get_article_keywords(self, pmid: list[str]) -> list[str]:
        """
        Retrieves keywords for a list of PMIDs.
        """
        articles = await self.fetch(pmid)

        if not articles:
            return []

        keywords = set()

        for article in articles:
            medline_citation = article.get("MedlineCitation", {})
            if medline_citation:
                # get mesh headings
                mesh_heading_list = medline_citation.get("MeshHeadingList", {})
                if mesh_heading_list:
                    mesh_headings = mesh_heading_list.get("MeshHeading", [])
                    if isinstance(mesh_headings, dict):
                        mesh_headings = [mesh_headings]

                    for mesh_heading in mesh_headings:
                        descriptor_name = mesh_heading.get("DescriptorName", {})
                        if descriptor_name:
                            keyword = descriptor_name.get("#text", "")
                            if keyword and descriptor_name.get("@MajorTopicYN") == "Y":
                                keywords.add(keyword)

                # get other keywords
                keyword_list = medline_citation.get("KeywordList", {})
                if keyword_list:
                    keywords_data = keyword_list.get("Keyword", [])
                    if isinstance(keywords_data, dict):
                        keywords_data = [keywords_data]

                    for keyword_entry in keywords_data:
                        keyword = keyword_entry.get("#text", "")
                        if keyword:
                            keywords.add(keyword)

        return list(keywords)

    @asynccontextmanager
    async def _create_http_client(self) -> AsyncIterator[httpx.AsyncClient]:
        if not self.tool_name or not self.email:
            raise ValueError("Tool name and email must be provided")

        headers = {
            "tool": self.tool_name,
            "email": self.email,
        }
        timeout = httpx.Timeout(30.0, connect=5.0)
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            yield client
