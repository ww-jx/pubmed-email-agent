from typing import cast
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, START, END

from src.pubmed_email_agent.agent.state import Summary
from src.pubmed_email_agent.agent.state import AgentState

from src.pubmed_email_agent.tools.llm.client import LLMTools
from src.pubmed_email_agent.tools.user.client import UserTools
from src.pubmed_email_agent.tools.pubmed.client import PubmedTools
from src.pubmed_email_agent.logger import get_logger

logger = get_logger(__name__)


class Agent:
    def __init__(
        self,
        user_tools: UserTools,
        llm_tools: LLMTools,
        pubmed_tools: PubmedTools,
        article_count: int = 5,
        max_retries: int = 3,
    ):
        self.user_tools = user_tools
        self.llm_tools = llm_tools
        self.pubmed_tools = pubmed_tools

        self.article_count = article_count
        self.max_retries = max_retries

        self.app = self._build_graph()

    async def run(self, initial_state: AgentState) -> AgentState:
        state = await self.app.ainvoke(
            initial_state,
            config={"run_name": f"user-{initial_state['user_id']}"},
        )

        return cast(AgentState, state)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # add nodes
        workflow.add_node("get_user_profile", self._get_user_data)
        workflow.add_node("process_feedback", self._process_feedback)
        workflow.add_node("generate_search_request", self._generate_search_request)
        workflow.add_node("search_for_articles", self._search_for_articles)
        workflow.add_node("fetch_article_details", self._fetch_article_details)
        workflow.add_node("summarize_articles", self._summarize_articles)
        workflow.add_node("format_email", self._format_email)
        workflow.add_node("send_email", self._send_email)

        # add edges
        workflow.add_edge(START, "get_user_profile")
        workflow.add_conditional_edges(
            "get_user_profile",
            self._subscription_check,
            {
                "continue": "process_feedback",
                "end": END,
            },
        )
        workflow.add_edge("process_feedback", "generate_search_request")
        workflow.add_edge("generate_search_request", "search_for_articles")

        workflow.add_conditional_edges(
            "search_for_articles",
            self._article_check,
            {
                "retry": "generate_search_request",
                "proceed": "fetch_article_details",
                "end": END,
            },
        )
        workflow.add_edge("fetch_article_details", "summarize_articles")
        workflow.add_edge("summarize_articles", "format_email")
        workflow.add_edge("format_email", "send_email")
        workflow.add_edge("send_email", END)

        return workflow.compile()

    def _subscription_check(self, state: AgentState) -> str:
        if state["user_profile"].subscribed:
            logger.info("User is subscribed, proceeding.")
            return "continue"

        return "end"

    def _article_check(self, state: AgentState) -> str:
        """
        Check if retrieved articles are sufficient
        """

        retries = state.get("retries", 0)

        if len(state["article_ids"]) >= self.article_count:
            logger.info("check passed: sufficient articles found")
            return "proceed"

        if retries < self.max_retries:
            logger.info("check failed: insufficient articles, retrying")
            return "retry"

        if retries >= self.max_retries and len(state["article_ids"]) == 0:
            logger.warning("No articles found after maximum retries, ending process")
            return "end"

        logger.info(
            "check failed: maximum retries reached, proceeding with available articles"
        )
        return "proceed"

    async def _get_user_data(self, state: AgentState) -> dict:
        logger.info(f"Fetching profile for user: {state['user_id']}")

        profile = await self.user_tools.get_user_profile(state["user_id"])

        if not profile:
            raise ValueError(f"No profile found for user ID: {state['user_id']}")

        # store user feedback
        user_feedback = await self.user_tools.get_user_feedback(state["user_id"])

        profile.feedback = user_feedback

        return {"user_profile": profile}

    async def _process_feedback(self, state: AgentState) -> dict:
        logger.info("Processing user feedback")

        profile = state["user_profile"]
        feedback = profile.feedback

        if not feedback:
            logger.info("No feedback to process")
            return {"article_ids": [], "negative_keywords": []}

        positive_pmids = [article.pmid for article in feedback if article.rating >= 4]

        negative_pmids = [article.pmid for article in feedback if article.rating <= 2]

        # get related articles for positive feedback
        related_articles = await self.pubmed_tools.get_related_articles(positive_pmids)

        # get negative keywords to avoid
        negative_keywords = await self.pubmed_tools.get_article_keywords(negative_pmids)

        logger.info(f"Found {len(related_articles)} related articles from feedback")

        return {
            "article_ids": related_articles,
            "negative_keywords": negative_keywords,
        }

    async def _generate_search_request(self, state: AgentState) -> dict:
        logger.info("Generating search request")

        profile = state["user_profile"]
        if profile.last_email_date is None:
            search_date = datetime.now().date() - timedelta(days=7)
        else:
            search_date = profile.last_email_date + timedelta(days=1)

        search_from_date_str = search_date.strftime("%Y/%m/%d")

        # Calculate how many more articles we need from search
        existing_count = len(state.get("article_ids", []))
        search_count = max(0, self.article_count - existing_count)

        logger.info(
            f"Requesting {search_count} articles from search (already have {existing_count} articles)"
        )

        previous_searches = state.get("previous_searches", [])

        search_req = await self.llm_tools.generate_search_query(
            profile.conditions,
            state["negative_keywords"],
            search_from_date_str,
            search_count,
            previous_searches,
        )

        search_req.retmax = search_count
        search_req.sort = "pub_date"

        return {"search_request": search_req}

    async def _search_for_articles(self, state: AgentState) -> dict:
        logger.info("Searching PubMed")

        search_ids, feedback = await self.pubmed_tools.search(state["search_request"])
        logger.info(
            f"Found {len(search_ids)} articles from search. PubMed Count: {feedback.get('count')}"
        )

        # Get existing article IDs and append new search results
        article_ids = list(dict.fromkeys(state.get("article_ids", []) + search_ids))[: self.article_count]

        logger.info(f"Total unique articles: {len(article_ids)}")

        retries = state.get("retries", 0)
        previous_searches = state.get("previous_searches", [])

        if len(article_ids) < self.article_count:
            retries += 1
            previous_searches.append(
                {
                    "attempted_query": state["search_request"].term,
                    "pubmed_feedback": feedback,
                }
            )

        return {
            "article_ids": article_ids,
            "retries": retries,
            "previous_searches": previous_searches,
        }

    async def _fetch_article_details(self, state: AgentState) -> dict:
        logger.info(
            f"Fetching article details for {len(state['article_ids'])} articles"
        )

        articles = await self.pubmed_tools.fetch(state["article_ids"])

        return {"fetched_articles": articles}

    async def _summarize_articles(self, state: AgentState) -> dict:
        logger.info("Summarizing articles")

        summaries = []

        articles_data = state["fetched_articles"]

        articles = [articles_data] if isinstance(articles_data, dict) else articles_data

        for article in articles:
            pmid = article.get("MedlineCitation", {}).get("PMID", "").get("#text", "")
            title = (
                article.get("MedlineCitation", {})
                .get("Article", {})
                .get("ArticleTitle", "")
            )
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            summary_text = await self.llm_tools.summarize_article(article)

            summary = Summary(
                pmid=pmid,
                title=title,
                link=link,
                summary=summary_text,
                rating_links_html="",
            )

            summaries.append(summary)

        return {"summaries": summaries}

    async def _format_email(self, state: AgentState) -> dict:
        logger.info("Formatting email content")

        user_profile = state["user_profile"]
        summaries = state["summaries"]

        email_content = await self.llm_tools.format_email(user_profile, summaries)

        return {"email_content": email_content}

    async def _send_email(self, state: AgentState) -> dict:
        logger.info("Sending email")

        user_profile = state["user_profile"]
        email_content = state["email_content"]

        subject = f"{user_profile.first_name}'s Weekly PubMed Digest"

        success = self.user_tools.send_email(user_profile.email, subject, email_content)

        if success:
            logger.info(f"Email sent to {user_profile.id}")
            update_success = await self.user_tools.update_last_email_date(
                user_profile.id
            )
            if not update_success:
                logger.error(
                    f"Failed to update last email date for user {user_profile.id}"
                )
        else:
            logger.error(f"Failed to send email to {user_profile.id}")

        return {}
