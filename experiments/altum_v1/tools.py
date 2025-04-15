import os
from openai import OpenAI

from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import re
from PIL import Image as PILImage
import glob

load_dotenv()

from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
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
    print(f"Perplexity usage: {response.usage}")
    
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
            6. Critically evaluate how well- or poorly-formed the plot is (see below)

            Rubric for evaluating plot quality:
| **Criteria**                  | **Good**                                                            | **Fair**                                                              | **Poor**                                                              |
|-------------------------------|---------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
| **Plot Type Appropriateness** | Plot type is well-suited to the data and clearly conveys its message. | Plot type is somewhat appropriate but may lead to minor confusion.    | Plot type is poorly chosen, causing significant misinterpretation.    |
| **Data Integrity & Accuracy** | Data are accurately represented with proper scaling and minimal errors. | Minor inaccuracies or scaling issues are present.                     | Data are significantly misrepresented or distorted.                   |
| **Clarity & Readability**     | All elements (labels, legends, etc.) are clear, legible, and organized.  | Some elements are hard to read or slightly cluttered.                   | The plot is cluttered with illegible or missing text elements.          |
| **Self-Containment & Utility**| Plot includes all necessary details (titles, labels, legends) for stand-alone understanding. | Key details are missing, requiring some effort to grasp the plot's intent. | Essential information is absent, leaving the viewer confused.         |
| **Overall Visual Quality**    | Clean design that focuses on clear data communication.               | Visual distractions are present but do not severely hinder understanding. | Distracting design elements significantly impair data communication.  |

            
            
            Your analysis should be thorough but concise, focusing on the most important aspects of the visualization. Give the score (Good, Fair, Poor) for each of the criteria above and explain why briefly. The goal is to help the plot creator understand how to improve the plot and also help readers to be aware of the flaws in the plot.
            
            If given specific questions about the plot, answer them directly and clearly.
            
            Your response should always be in this format:

            **OVERALL EXPLANATION OF PLOT**
            [OVERALL EXPLANATION OF PLOT with analysis of what the plot shows]
            [For this part, try to describe the plot in a way that a blind person can understand what the plot shows and what it is trying to communicate]
            [Try to estimate the range of values in the plot, if possible]

            **CRITIQUE OF PLOT**
            [CRITIQUE OF PLOT with score (Good, Fair, Poor) for each criterion and explanation]

            **QUESTIONS AND ANSWERS ABOUT THE PLOT**
            Question: [QUESTION]
            Answer: [ANSWER]
            
            """,
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

        # Get the response - directly await the agent's response without Console wrapper
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

async def search_directory(directory_path: str, pattern: str = None, recursive: bool = False) -> str:
    """
    Search for files in a specified directory, optionally filtering by pattern and searching recursively.
    
    Args:
        directory_path: Path to the directory to search
        pattern: Optional glob pattern to filter results (e.g., "*.png" for all PNG files)
        recursive: Whether to search subdirectories recursively
        
    Returns:
        A string containing the list of matching files and their details
    """
    try:
        # Ensure the directory exists
        if not os.path.exists(directory_path):
            return f"Error: Directory '{directory_path}' does not exist."
        
        if not os.path.isdir(directory_path):
            return f"Error: '{directory_path}' is not a directory."
            
        # Construct search pattern
        search_path = os.path.join(directory_path, pattern or "*")
        
        # Find files matching the pattern
        if recursive:
            # For recursive search
            matches = []
            if pattern:
                for root, _, _ in os.walk(directory_path):
                    matches.extend(glob.glob(os.path.join(root, pattern)))
            else:
                for root, _, files in os.walk(directory_path):
                    matches.extend([os.path.join(root, file) for file in files])
        else:
            # For non-recursive search
            matches = glob.glob(search_path)
        
        # Sort results
        matches.sort()
        
        # Format the output
        if not matches:
            if pattern:
                return f"No files matching '{pattern}' found in '{directory_path}'."
            else:
                return f"No files found in '{directory_path}'."
        
        result = f"Found {len(matches)} files in '{directory_path}'"
        if pattern:
            result += f" matching '{pattern}'"
        result += ":\n\n"
        
        # Add file details
        for filepath in matches:
            filename = os.path.basename(filepath)
            size = os.path.getsize(filepath)
            mod_time = os.path.getmtime(filepath)
            
            # Format size in human-readable format
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size/(1024*1024):.1f} MB"
                
            # Format modified time
            import datetime
            mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to result
            result += f"- {filename} ({size_str}, modified: {mod_time_str})\n"
            
        return result
        
    except Exception as e:
        return f"Error searching directory: {str(e)}"


async def read_text_file(filepath: str) -> str:
    """
    Read the contents of a text file.
    """
    try:
        with open(filepath, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Error reading text file: {str(e)}"


async def read_arrow_file(filepath: str) -> str:
    """
    Read the contents of an Arrow file (feather format).
    """
    import pandas as pd
    
    try:
        return pd.read_feather(filepath)
    except Exception as e:
        return f"Error reading Arrow file: {str(e)}"
