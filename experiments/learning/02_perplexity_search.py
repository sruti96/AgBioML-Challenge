## For this example, we are going to use the perplexity MCP server to enable LLM researchers

from openai import OpenAI
import asyncio

from dotenv import load_dotenv
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import StructuredMessage

import requests
from bs4 import BeautifulSoup
import re

load_dotenv()


YOUR_API_KEY = os.getenv("PERPLEXITY_API_KEY")

async def query_perplexity(query: str) -> tuple[str, list[str]]:

    model = "sonar"
    # model_options = ['sonar', 'sonar-pro', 'sonar-deep-research', 'sonar-reasoning-pro', 'sonar-reasoning']
    # assert model in model_options, f"Model must be one of the following: {model_options}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {   
            "role": "user",
            "content": (
                f"{query}"
            ),
        },
    ]

    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

    # chat completion without streaming
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    
    # Format the response
    response_dict = response.model_dump()
    content= response_dict["choices"][0]["message"]["content"]
    citations = response_dict["citations"]
    # Format the citations so that [citation_id] -> [citation_text]
    citations_dict = {id + 1: citation for id, citation in enumerate(citations)}
    # Add the sources to the content
    content = f"{content}\n\nSources:\n"
    for id, citation in citations_dict.items():
        content += f"{id}. {citation}\n"

    return content, citations_dict


async def extract_citation_from_perplexity(citation_url: str) -> str:
    """
    Fetch and parse content from a citation URL provided by Perplexity.
    
    Args:
        citation_url: URL of the citation to fetch
        max_length: Maximum length of the returned summary
        
    Returns:
        A string containing the extracted content or error message
    """
    max_length = 100_000
    
    try:
        # Extract the actual URL if it's embedded in the citation string
        # URLs in Perplexity citations are often formatted like "title - source (url)"
        url_match = re.search(r'https?://[^\s)]+', citation_url)
        if url_match:
            url = url_match.group(0)
        else:
            url = citation_url
            
        # Fetch the webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text and clean it up
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate to max_length with a note if needed
        if len(text) > max_length:
            return text[:max_length] + " [text truncated due to length]"
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching citation: {str(e)}"
    except Exception as e:
        return f"Error processing citation: {str(e)}"




if __name__ == "__main__":
    # try:
    #     content, citations = asyncio.run(query_perplexity("What transcription factors might be able to reverse the age of cells and tissues? How would these be tested experimentally?"))
    #     print(content)
    #     print(citations)
    # except Exception as e:
    #     raise Exception(f"Error querying perplexity: {str(e)}")

    # # Now, for citation 1, let's query perplexity to learn more about it and why it is relevant to the query
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # citation_1 = citations[1]


    # # Use the basic extraction function
    # print("BASIC EXTRACTION:")
    # summary_of_citation = extract_citation_from_perplexity(citation_1)
    # print(summary_of_citation)
    
    # # Use the agent-based extraction function
    # print("\n\nAGENT-BASED EXTRACTION:")
    # agent_summary = asyncio.run(extract_citation_with_agent(citation_1))
    # print(agent_summary)


    # Finally, let's do an example where an agent calls the perplexity tool
    model_client = OpenAIChatCompletionClient(
        model='gpt-4o'
    )

    from pydantic import BaseModel
    from autogen_agentchat.messages import StructuredMessage


    class OutputMessageContent(BaseModel):
        user_query: str
        perplexity_response: str
        citation_summaries: list[str]

    # OutputMessage = StructuredMessage(
    #     content=OutputMessageContent,
    #     source="researcher_bot"
    # )

    from autogen_core.tools import FunctionTool

    query_perplexity_tool = FunctionTool(query_perplexity, strict=True, description="Query perplexity for information on a research topic")
    extract_citation_from_perplexity_tool = FunctionTool(extract_citation_from_perplexity, strict=True, description="Extract a summary of a citation from perplexity")

    agent = AssistantAgent(
        name='researcher_bot',
        model_client=model_client,
        tools=[query_perplexity_tool, extract_citation_from_perplexity_tool],
        system_message="""You are an AI Research Assistant.
        Your goal is to help the user research scientific topics they are interested in.
        You will take a user question and then:
        1. Formulate the question as a well-structured research query for perplexity
        2. Query perplexity AI using the perplexity query tool
        3. Analyze the response from perplexity.
        4. Follow up on sources of interest as needed using the extract_citation tool.
        5. Provide a well-written and precise summary of the research results to the user.

        The output format is, in JSON:
        {
            'user_query: <original_user_query>,
            'perplexity_response': <response_from_perplexity>,
            'citation_summaries': [
                {
                    'citation_url': <citation_url>,
                    'citation_summary': <summary_of_citation>
                }
            ],
            'final_summary': <final_summary_of_research>
        }
""",
        model_client_stream=True,
        reflect_on_tool_use=True,
        output_content_type=OutputMessageContent

    )

    async def main() -> None:
        await Console(agent.run_stream(task="What transcription factors might be able to reverse the age of cells and tissues? How would these be tested experimentally?"))
        await model_client.close()

    asyncio.run(main())

    



