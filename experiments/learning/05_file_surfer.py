import asyncio
import os
from dotenv import load_dotenv

from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

load_dotenv()

async def main():
    # Create a simple text file to read
    with open('test.txt', 'w') as f:
        f.write("""This is a test file.
It contains multiple lines of text.
Each line has different content.
The last line is here.""")

    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o"
    )

    # Create the FileSurfer agent
    file_surfer = FileSurfer(
        name="file_reader",
        model_client=model_client,
        base_path=os.getcwd()  # Use current working directory
    )

    # Create a proper message object
    message = TextMessage(
        content="Read the contents of age_distribution.png",
        source="user"
    )

    # Read the file
    response = await file_surfer.on_messages(
        messages=[message],
        cancellation_token=None
    )
    breakpoint()

    # Print the response
    print("\nFile contents:")
    print(response.chat_message.content)

    # Close the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main()) 