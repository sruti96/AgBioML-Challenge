import asyncio
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_ext.agents.file_surfer import FileSurfer

# Import our custom tool
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

load_dotenv()

def generate_random_plot():
    """Generate a random plot of age distribution."""
    # Generate random age data
    np.random.seed(42)  # For reproducibility
    ages = np.random.normal(loc=50, scale=15, size=1000)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(ages, bins=30, edgecolor='black')
    plt.title('Age Distribution in Study Population')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('age_distribution.png')
    plt.close()
    
    print("Generated age_distribution.png")

async def main():
    # First generate the plot
    generate_random_plot()
    
    # Initialize the model client with a vision-capable model
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
        
        Your analysis should be thorough but concise, focusing on the most important aspects of the visualization.""",
        model_client_stream=True
    )

    # Load the saved plot as a PIL Image
    plot_image = PILImage.open('age_distribution.png')

    # Create a multimodal message with both text and the plot image
    message = MultiModalMessage(
        content=[
            "Please analyze this plot and describe what it shows:",
            Image(plot_image)
        ],
        source="user"
    )

    # Get the response
    response = await agent.on_messages(
        messages=[message],
        cancellation_token=None
    )

    # Print the response
    print("\nPlot Analysis:")
    print(response.chat_message.content)

    # Close the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main()) 