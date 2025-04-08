import os
from openai import OpenAI

from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import re
from PIL import Image as PILImage

load_dotenv()

from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_agentchat.agents import AssistantAgent

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


async def analyze_plot_file(filepath: str, prompt: str | None = None) -> str:
    """
    Analyze a plot file and return a description of its contents.
    
    Args:
        filepath: Path to the plot file (relative to the working directory)
        prompt: Optional custom prompt to ask about the plot. If None, uses a default prompt.
        
    Returns:
        A string containing the analysis of the plot or an error message
    """
    try:
        # Initialize the model client
        model_client = OpenAIChatCompletionClient(
            model="gpt-4o"
        )

        # Create the agent that can handle multimodal input
        agent = AssistantAgent(
            name="plot_analyzer",
            model_client=model_client,
            system_message="""You are an agent that can analyze and describe plots.
            When given a plot, you should:
            1. Describe what type of plot it is
            2. Explain what the plot shows
            3. Identify key features of the distribution
            4. Note any interesting patterns or outliers
            5. Provide a clear summary of the insights
            
            Your analysis should be thorough but concise, focusing on the most important aspects of the visualization.
            
            If given specific questions about the plot, answer them directly and clearly.""",
            model_client_stream=True
        )

        # Load the plot as a PIL Image
        plot_image = PILImage.open(filepath)

        # Use default prompt if none provided
        if prompt is None:
            prompt = """Please analyze this plot and describe what it shows. Focus on:
            1. The type of plot and its purpose
            2. The distribution characteristics and patterns
            3. Any notable outliers or anomalies
            4. The approximate range of values
            5. Any insights that could be relevant for data analysis"""

        # Create a multimodal message with both text and the plot image
        message = MultiModalMessage(
            content=[
                prompt,
                Image(plot_image)
            ],
            source="user"
        )

        # Get the response
        response = await agent.on_messages(
            messages=[message],
            cancellation_token=None
        )

        # Close the model client
        await model_client.close()

        # Return the analysis
        return response.chat_message.content
        
    except Exception as e:
        return f"Error analyzing plot file: {str(e)}"

# Create the tool instance
analyze_plot_tool = FunctionTool(
    analyze_plot_file,
    description="""Analyze a plot file and return a description of its contents.
    You can provide a custom prompt to ask specific questions about the plot.
    If no prompt is provided, a default analysis will be performed.""",
    name="analyze_plot"
)
