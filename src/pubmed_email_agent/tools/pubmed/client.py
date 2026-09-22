import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import xmltodict
from pubmedclient.models import (
    Db,
    EFetchRequest,
    ELinkCmd,
    ELinkRequest,
    ESearchRequest,
    RetMode,
)
from pubmedclient.sdk import efetch, elink, esearch

from src.pubmed_email_agent.logger import get_logger

logger = get_logger(__name__)


class PubmedTools:
    def __init__(self, tool_name: str, email: str):
        self.tool_name = tool_name
        self.email = email

        self.db = Db.PUBMED

    async def search(self, params: ESearchRequest) -> tuple[list[str], dict]:
        """
        Searches PubMed with the given parameters and returns a list of article ids.
        """
        params.db = self.db.value

        try:
            async with self._create_http_client() as client:
                response = await esearch(client, params)
                result = response.esearchresult

                result_dict = (
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else vars(result)
                )

                feedback = {
                    "count": result_dict.get("count", "0"),
                    "errorlist": result_dict.get("errorlist", None),
                    "warninglist": result_dict.get("warninglist", None),
                    "querytranslation": result_dict.get("querytranslation", None),
                }

                return result.idlist, feedback

        except Exception as e:
            logger.error(f"PubMed API Search Error: {e}")

            # inject network error into feedback dict
            fallback_feedback = {
                "count": "0",
                "errorlist": {"network_error": f"API Request Failed: {str(e)}"},
                "warninglist": None,
                "querytranslation": None,
            }
            return [], fallback_feedback

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

        try:
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
        except Exception as e:
            logger.error(f"PubMed API Fetch Error: {e}")
            return []

    def parse_article(self, article: dict) -> dict:
        """parses xmltodict PubMed response into dict"""
        pmid = article.get("MedlineCitation", {}).get("PMID", {})
        pmid_str = (
            pmid.get("#text", "")
            if isinstance(pmid, dict)
            else str(pmid)
            if pmid
            else ""
        )

        title = (
            article.get("MedlineCitation", {})
            .get("Article", {})
            .get("ArticleTitle", "")
        )

        if isinstance(title, dict):
            title = title.get("#text", "")

        abstract_raw = (
            article.get("MedlineCitation", {})
            .get("Article", {})
            .get("Abstract", {})
            .get("AbstractText", "")
        )

        if isinstance(abstract_raw, list):
            abstract = " ".join(
                [a.get("#text", "") for a in abstract_raw if isinstance(a, dict)]
            )
        elif isinstance(abstract_raw, dict):
            abstract = abstract_raw.get("#text", "")
        else:
            abstract = str(abstract_raw) if abstract_raw else ""

        return {
            "pmid": pmid_str,
            "title": str(title) if title else "",
            "abstract": abstract,
        }

    async def get_related_articles(
        self, pmids: list[str], search_window: tuple[str, str], article_count: int = 2
    ) -> list[str]:
        """
        Given a list of PMIDs, retrieves the single most related article PMID for each.

        Returns a dictionary mapping each input PMID to its single most related article PMID
        (or None if none found).
        """
        if not pmids:
            return []

        mindate, maxdate = search_window

        params = ELinkRequest(
            dbfrom=Db.PUBMED,
            db=Db.PUBMED,
            id=pmids,
            cmd=ELinkCmd.NEIGHBOR_SCORE,
            retmode=RetMode.JSON,
            mindate=mindate,
            maxdate=maxdate,
            datetype="edat",
        )

        try:
            async with self._create_http_client() as client:
                response = await elink(client, params)
                data = json.loads(response, strict=False)
        except Exception as e:
            logger.error(f"Error fetching related articles: {e}")
            return []

        scored: list[tuple[float, str]] = []

        try:
            for linkset in data.get("linksets", []):
                for linksetdb in linkset.get("linksetdbs", []):
                    if linksetdb.get("linkname") != "pubmed_pubmed":
                        continue

                    for link in linksetdb.get("links", []):
                        pmid = link.get("id")
                        if not pmid:
                            continue

                        try:
                            score = float(link.get("score", 0.0))
                        except (ValueError, TypeError):
                            score = 0.0

                        scored.append((score, str(pmid)))
        except Exception as e:
            logger.error(f"Error parsing related articles data: {e}")
            return []

        # highest score first, then dedupe
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = list(dict.fromkeys(pmid for _, pmid in scored))

        return ranked[:article_count]

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
