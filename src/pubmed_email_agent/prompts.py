GENERATE_QUERY_SYS = """
You are a helpful assistant that helps to generate PubMed search queries.
Based on the user's interests and preferences, generate the search parameters
"""
GENERATE_QUERY_USER = """
Generate the search parameters based on the following details: \n
- Interests: {interests}\n
- Only retrieve results published after {date}\n
"""

SUMMARIZE_ARTICLE_SYS = """
You are a helpful assistant that helps to summarize PubMed articles.
Based on the article data, generate a concise summary that is easy to understand.
Only return a summary paragraph. Do not include any other text, such as author.
"""
SUMMARIZE_ARTICLE_USER = """
Summarize the following article data: \n" "{article_data}\n
"""

FORMAT_EMAIL_SYS = """
You are an expert medical writer. Your task is to create a personalized weekly email summary
of recent clinical research for a user. The tone should be friendly, informative, and clear.

Use language representing an organization, not an individual.
- Use an impersonal voice or "we".
- AVOID using "I" or personal anecdotes.

The output must be only well-structured Markdown format.
Do not include a subject line or any other text outside the email body.
Always include a disclaimer that this is not medical advice.
Sign off the email with "GiraffeGuru Team".
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
        - At the end of each article section, include a rating section using the HTML from the `rating_links_html` field. \n
3.  **Analysis & PCP Questions:** \n
    - After the article sections, create general sections for 'Prep for Your PCP' and 'Smart Questions to Ask', based on the user's profile and the findings from the articles. \n
"""
