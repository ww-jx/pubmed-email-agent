from typing import cast
from langgraph.graph import StateGraph, START, END
from sqlalchemy.sql.functions import user

from src.pubmed_email_agent.agent.state import Summary
from src.pubmed_email_agent.agent.state import AgentState

from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools
from src.pubmed_email_agent.tools.user.client import UserTools


class Agent:
    def __init__(
        self, user_tools: UserTools, llm_tools: LLMTools, pubmed_tools: PubmedTools
    ):
        self.user_tools = user_tools
        self.llm_tools = llm_tools
        self.pubmed_tools = pubmed_tools

        self.app = self._build_graph()

    async def run(self, initial_state: AgentState) -> AgentState:
        state = await self.app.ainvoke(initial_state)

        return cast(AgentState, state)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # add nodes
        workflow.add_node("get_user_profile", self._get_user_profile)
        workflow.add_node("generate_search_request", self._generate_search_request)
        workflow.add_node("search_for_articles", self._search_for_articles)
        workflow.add_node("fetch_article_details", self._fetch_article_details)
        workflow.add_node("summarize_articles", self._summarize_articles)
        workflow.add_node("format_email", self._format_email)
        workflow.add_node("send_email", self._send_email)

        # add edges
        workflow.add_edge(START, "get_user_profile")
        workflow.add_edge("get_user_profile", "generate_search_request")
        workflow.add_edge("generate_search_request", "search_for_articles")
        workflow.add_edge("search_for_articles", "fetch_article_details")
        workflow.add_edge("fetch_article_details", "summarize_articles")
        workflow.add_edge("summarize_articles", "format_email")
        workflow.add_edge("format_email", "send_email")
        workflow.add_edge("send_email", END)

        return workflow.compile()

    async def _get_user_profile(self, state: AgentState) -> dict:
        print(f"Fetching profile for user: {state['user_id']}")

        profile = await self.user_tools.get_user_profile(state["user_id"])

        if not profile:
            raise ValueError(f"No profile found for user ID: {state['user_id']}")

        return {"user_profile": profile}

    def _generate_search_request(self, state: AgentState) -> dict:
        print("Generating search request")

        profile = state["user_profile"]

        search_req = self.llm_tools.generate_search_query(
            profile.conditions, state["search_from_date"]
        )

        return {"search_request": search_req}

    async def _search_for_articles(self, state: AgentState) -> dict:
        print("Searching PubMed")

        ids = await self.pubmed_tools.search(state["search_request"])

        print(f"Found {len(ids)} articles.")

        return {"article_ids": ids}

    async def _fetch_article_details(self, state: AgentState) -> dict:
        print("Fetching article details")

        articles = await self.pubmed_tools.fetch(state["article_ids"])

        return {"fetched_articles": articles}

    def _summarize_articles(self, state: AgentState) -> dict:
        print("Summarizing articles")

        summaries = []

        articles = state["fetched_articles"]

        for article in articles:
            pmid = article.get("MedlineCitation", {}).get("PMID", "").get("#text", "")
            title = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("ArticleTitle", "")
            )
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            summary_text = self.llm_tools.summarize_article(article)

            summary = Summary(
                pmid=pmid,
                title=title,
                link=link,
                summary=summary_text,
            )

            summaries.append(summary)

        return {"summaries": summaries}

    def _format_email(self, state: AgentState) -> dict:
        print("Formatting email content")

        user_profile = state["user_profile"]
        summaries = state["summaries"]

        email_content = self.llm_tools.format_email(user_profile, summaries)

        return {"email_content": email_content}

    def _send_email(self, state: AgentState) -> dict:
        print("Sending email")

        user_profile = state["user_profile"]
        email_content = state["email_content"]

        subject = f"{user_profile.first_name}'s Weekly PubMed Digest"

        success = self.user_tools.send_email(user_profile.email, subject, email_content)

        if success:
            print(f"Email sent to {user_profile.id}")
        else:
            print(f"Failed to send email to {user_profile.id}")

        return {}
