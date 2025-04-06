## Create a basic assistant

from dotenv import load_dotenv
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# Define the model client.
model_client = OpenAIChatCompletionClient(
    model = "gpt-4o"
)

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 21 degrees and sunny."


# Define assistant agent with model, tool, system message, and reflection upon tool usage
# TODO: What is reflection on tool use?
agent = AssistantAgent(
    name='weather_agent',
    model_client=model_client,
    tools=[get_weather],
    system_message='You are a helpful assistant.',
    reflect_on_tool_use=True,
    model_client_stream=True
)


# Run the agent and stream messages
async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York City?"))
    # Close the connection
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())






