import os
from openai import OpenAI

from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import re

load_dotenv()

from autogen_ext.agents.file_surfer import FileSurfer
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient


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


async def format_webpage(url: str) -> str:
    """
    Fetch and parse HTML content from a URL.
    
    Args:
        url: URL of the citation to fetch
        
    Returns:
        A string containing the extracted content or error message
    """
    max_length = 100_000
    
    try:
        # Extract the actual URL if it's embedded in the citation string
        # URLs in Perplexity citations are often formatted like "title - source (url)"
        url_match = re.search(r'https?://[^\s)]+', url)
        if url_match:
            url = url_match.group(0)

            
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

async def read_plot_file(filepath: str) -> str:
    """
    Read and return the contents of a plot file.
    
    Args:
        filepath: Path to the plot file (relative to the working directory)
        
    Returns:
        A string containing the plot file contents or an error message
    """
    try:
        # Initialize FileSurfer agent
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        file_surfer = FileSurfer(
            name="plot_reader",
            model_client=model_client,
            base_path=os.getcwd()  # Use current working directory
        )
        
        # Read the file
        response = await file_surfer.on_messages(
            messages=[{"role": "user", "content": f"Read the contents of {filepath}"}],
            cancellation_token=None
        )
        breakpoint()
        
        # Return the file contents
        return response.chat_message.content
        
    except Exception as e:
        return f"Error reading plot file: {str(e)}"

# Create the tool instance
read_plot_tool = FunctionTool(
    read_plot_file,
    description="Read the contents of a plot file and return its contents as text",
    name="read_plot"
)
