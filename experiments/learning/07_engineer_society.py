# In this example, we implement a hierarchical agent system with the following structure:
# RoundRobin(RoundRobin(engineer, executor), RoundRobin(critic))
# 
# This creates an "EngineerSociety" that:
# 1. Has an inner team where an 'engineer' agent writes code and the 'executor' agent runs it
# 2. Has a 'critic' agent that reviews the results and provides feedback
# 3. Cycles between the engineer team and critic until the critic approves the work
#
# The task demonstrates calculating moving averages for S&P500 data, but this pattern
# could be used for any task requiring code writing, execution, and review.

import asyncio
import os
from typing import Sequence

from autogen_core.agent import BaseChatAgent, CancellationToken
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_core.messages import Response
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

# System prompts for our agents
ENGINEER_SYSTEM_PROMPT = """
You are an expert Python engineer who specializes in data analysis and visualization.

Your task is to write Python code to calculate and visualize moving averages of the S&P500 from 2016 to 2018, 
and identify the point with the highest moving average.

Follow these steps in your implementation:
1. Use yfinance package to get S&P500 data (ticker: ^GSPC) for the period 2016-01-01 to 2018-12-31
2. Calculate multiple moving averages (20-day, 50-day, and 100-day)
3. Visualize the stock price and all three moving averages on the same chart with different colors
4. Find and mark the point with the highest moving average for each window size
5. Save the visualization to a file named "sp500_moving_averages.png"
6. Print the date and value of the highest moving average for each window

Your code should be well-structured with:
- Clear import statements at the top
- Helpful comments explaining key sections
- Error handling for data retrieval
- Proper labeling on the visualization (title, axes, legend)
- Clear output of the highest moving average points

If you need to install any packages, use pip or conda commands as needed.
If the code executor reports any errors, fix them in your next response.

When you think your implementation is complete, indicate with "ENGINEER_DONE".
"""

CRITIC_SYSTEM_PROMPT = """
You are an expert Python code reviewer who specializes in data analysis and visualization.

Your task is to review the code and output produced by the engineer. Examine the implementation
for calculating and visualizing moving averages of the S&P500 from 2016 to 2018, and identifying
the point with the highest moving average.

Specifically check for:
1. Correct data retrieval using yfinance (ticker: ^GSPC) for the period 2016-01-01 to 2018-12-31
2. Proper calculation of moving averages (20-day, 50-day, and 100-day)
3. Clear visualization with all three moving averages plotted with different colors
4. Accurate identification and visible marking of the highest moving average points
5. Properly saved visualization as "sp500_moving_averages.png"
6. Clear output of dates and values for highest moving averages

Look for these potential issues:
- Missing or incorrect date range
- Calculation errors in moving averages
- Poor visualization (missing labels, legend, or unclear markings)
- Incorrect identification of maximum points
- Missing error handling
- Inefficient code

If you find any issues or areas for improvement, provide specific, constructive feedback.
If the implementation meets all requirements, indicate approval with "CRITIC_APPROVE".

Be thorough but fair in your assessment.
"""

class EngineerSociety(BaseChatAgent):
    def __init__(self, name: str, engineer_team: RoundRobinGroupChat, critic_team: RoundRobinGroupChat, critic_approve_token: str) -> None:
        super().__init__(name, description="An agent that performs data analysis with feedback from a critic.")
        self._engineer_team = engineer_team
        self._critic_team = critic_team
        self._critic_approve_token = critic_approve_token

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages through the engineer team and critic team until approved.
        
        The flow is:
        1. Engineer team (engineer + executor) writes and runs code
        2. Critic team reviews the results
        3. If critic approves, process ends
        4. If critic has feedback, engineer team gets the critic's feedback and revises
        """
        while True:
            # Run the first team with the given messages and returns the last message produced by the team.
            result_engineer = await self._engineer_team.run(task=messages, cancellation_token=cancellation_token)
            # To stream the inner messages, implement `on_messages_stream` and use that to implement `on_messages`.
            assert isinstance(result_engineer.messages[-1], TextMessage)
            messages = result_engineer.messages

            # Run the second team with the given messages and returns the last message produced by the team.
            result_critic = await self._critic_team.run(task=messages, cancellation_token=cancellation_token)
            # To stream the inner messages, implement `on_messages_stream` and use that to implement `on_messages`.
            assert isinstance(result_critic.messages[-1], TextMessage)

            if self._critic_approve_token in result_critic.messages[-1].content:
                break
            
        return Response(chat_message=result_engineer.messages[-1], inner_messages=result_engineer.messages[len(messages):-1])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        # Reset the inner team.
        await self._engineer_team.reset()
        await self._critic_team.reset()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)


async def main():
    # Create output directory for the code executor
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model client
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create the engineer agent
    engineer_agent = AssistantAgent(
        name="Engineer",
        system_message=ENGINEER_SYSTEM_PROMPT,
        model_client=model_client,
        model_client_stream=True
    )
    
    # Create the code executor agent with a timeout of 300 seconds (5 minutes)
    code_executor = LocalCommandLineCodeExecutor(
        timeout=300,
        work_dir=output_dir
    )
    code_executor_agent = CodeExecutorAgent(
        name="CodeExecutor",
        code_executor=code_executor
    )
    
    # Create the critic agent
    critic_agent = AssistantAgent(
        name="Critic",
        system_message=CRITIC_SYSTEM_PROMPT,
        model_client=model_client,
        model_client_stream=True
    )
    
    # Define termination conditions
    engineer_termination = TextMentionTermination("ENGINEER_DONE")
    critic_termination = TextMentionTermination("CRITIC_APPROVE")
    
    # Create the engineer team (engineer + code executor)
    engineer_team = RoundRobinGroupChat(
        participants=[engineer_agent, code_executor_agent],
        termination_condition=engineer_termination,
        max_turns=20
    )
    
    # Create the critic team (just the critic for now)
    critic_team = RoundRobinGroupChat(
        participants=[critic_agent],
        termination_condition=critic_termination,
        max_turns=5
    )
    
    # Create the engineer society
    engineer_society = EngineerSociety(
        name="engineer_society",
        engineer_team=engineer_team,
        critic_team=critic_team,
        critic_approve_token="CRITIC_APPROVE"
    )
    
    # Define the task message
    task_message = TextMessage(
        content="Calculate and visualize the moving averages (20-day, 50-day, and 100-day) of the S&P500 index between 2016 and 2018. Identify and mark the point with the highest moving average for each window. Save the visualization as 'sp500_moving_averages.png' and report the dates and values of these highest points.",
        source="User"
    )
    
    # Run the engineer society
    print(f"Starting EngineerSociety execution...")
    print(f"Output will be saved to: {output_dir}")
    response = await engineer_society.on_messages([task_message], CancellationToken())
    print("\nFinal result:")
    if isinstance(response.chat_message, TextMessage):
        print(response.chat_message.content)
    else:
        print("Unexpected response type")
    
    # Check if the output file exists
    output_file = os.path.join(output_dir, "sp500_moving_averages.png")
    if os.path.exists(output_file):
        print(f"\nVisualization saved successfully at: {output_file}")
    else:
        print("\nWarning: Visualization file not found")

if __name__ == "__main__":
    asyncio.run(main())

