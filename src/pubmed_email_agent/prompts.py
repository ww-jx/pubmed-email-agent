GENERATE_QUERY_SYS = """
You are a helpful assistant that helps to generate PubMed search queries.
Based on the user's interests and preferences, generate the search parameters
"""
GENERATE_QUERY_USER = """
Generate the search parameters based on the following details: \n
- Interests: {interests}\n
- Only retrieve results published after {date}\n
- The query should only return at most {article_count} articles
"""

SUMMARIZE_ARTICLE_SYS = """
You are a helpful assistant that helps to summarize PubMed articles.
Based on the article data, generate a concise summary that is easy to understand.
"""
SUMMARIZE_ARTICLE_USER = """
Summarize the following article data: \n" "{article_data}\n
"""

FORMAT_EMAIL_SYS = """
You are an expert medical writer. Your task is to create a personalized weekly email summary
of recent clinical research for a user. The tone should be friendly, informative, and clear.
The output must be in well-structured Markdown format.
Always include a disclaimer that this is not medical advice.
"""
FORMAT_EMAIL_USER = """
Please create the email using the following information:\n\n
**User Profile:**\n
{user_profile}\n\n
**Summarized Articles:**\n
{summaries_json}\n\n
**Email Structure Requirements:**\n
1.  **TL;DR:** \n
- Analyze all the provided articles and create a high-level, one-line summary for each of the user's conditions.\n
2.  **Detailed Article Sections:** \n"
    - Create a separate, dedicated section for EACH article provided in the summaries. Each section must include:\n
        - the article's full Title \n
        - its PMID \n
        - its Link \n
        - its full Summary\n
        - ELI5 section. This should still explain the article in detail, but made to be easy to understand. \n
3.  **Analysis & PCP Questions:** \n
    - After the article sections, create general sections for 'Prep for Your PCP' and 'Smart Questions to Ask', based on the user's profile and the findings from the articles. \n
4.  **Link Index:** \n
    - Create a final, clean list of all PubMed links.
"""
