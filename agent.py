from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
import os
from dotenv import load_dotenv
import arxiv
from typing import List, Dict
import asyncio
load_dotenv()

def arxiv_query(query: str, max_results: int = 5) -> List[Dict]:
    """ Return a compact list of arXiv papers matching the query. 
    
    Each element contains : ``title``, ``authors``, ``summary``, ``published``, and ``pdf_url``.
    The helper is wrapped as as Autogen *FunctionTool* below so it can be invoked by
    agents through normal tool use mechanisms"""


    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers: List[Dict] = []


    for result in client.results(search):
        papers.append({
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'published': result.published.strftime("%Y-%m-%d"),
            'pdf_url': result.pdf_url
        })
    return papers


model_client = OpenAIChatCompletionClient(
    model='gemini-2.5-flash',
    api_key=os.getenv('GEMINI_API_KEY')
    )

arxiv_research_agent = AssistantAgent(
    name="arxiv_research_agent",
    description='Create arXiv queries and retrieve candidate papers',
    model_client=model_client,
    tools= [arxiv_query],
    system_message = (
        "You are an AI assistant tasked with retrieving relevant academic papers from arXiv. "
        "When given a user topic and a requested number of papers, do the following:\n"
        "1. Generate the most effective arXiv query based on the user topic.\n"
        "2. Use the provided tool to fetch *five times* the number of papers requested to ensure a broader selection.\n"
        "3. From the retrieved papers, down-select the most relevant ones.\n"
        "4. Choose exactly the number of papers the user requested.\n"
        "5. Format the selected papers as concise JSON and send them to the summarizer component.\n"
        "Always prioritize paper relevance when selecting the final set."
    )
)

summarizer_agent = AssistantAgent(
    name='paper_summarizer',
    description='An agent that summarizes research papers.',
    model_client=model_client,
    system_message = (
        "You are an expert research assistant. When given a JSON list of research papers, "
        "generate a concise literature review in Markdown format:\n\n"
        "1. Begin with a 2–3 sentence introduction that provides context on the overall research topic.\n"
        "2. For each paper, add a bullet point with the following details:\n"
        "   - **Title** (as a Markdown link to the paper)\n"
        "   - **Authors**\n"
        "   - The specific **problem addressed**\n"
        "   - The paper’s **key contribution**\n"
        "3. End with a single-sentence takeaway summarizing trends or insights across the papers.\n"
    )
)

team = RoundRobinGroupChat(
    participants=[arxiv_research_agent, summarizer_agent],
    max_turns=2
)


async def run_team():
    task = "conduct a literature review on the latest advancements in quantum computing"

    async for msg in team.run_stream(task=task):
        print(msg)


if __name__ == "__main__":
    asyncio.run(run_team())