import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from PIL import Image as PILImage, ImageDraw

load_dotenv()

async def main():
    # Initialize the model client with a vision-capable model
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o"  # Using a vision-capable model
    )

    # Create an assistant agent that can handle multimodal input
    assistant = AssistantAgent(
        name="multimodal_assistant",
        model_client=model_client,
        system_message="""You are a helpful assistant that can understand both text and images.
        When given an image, you should describe what you see and answer any questions about it.
        When given text, you should respond appropriately to the text content.""",
        model_client_stream=True
    )

    # Create a test image
    img = PILImage.new('RGB', (200, 200), color='red')
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Test Image", fill='white')

    # Create a multimodal message with both text and image
    message = MultiModalMessage(
        content=[
            "Please describe this image and tell me what color it is:",
            Image(img)  # Pass the PIL Image object directly
        ],
        source="user"
    )

    # Get the response
    response = await assistant.on_messages(
        messages=[message],
        cancellation_token=None
    )

    # Print the response
    print("\nAssistant's response:")
    print(response.chat_message.content)

    # Close the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main()) 
