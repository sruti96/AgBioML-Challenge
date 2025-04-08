import asyncio
import yaml

from dotenv import load_dotenv
import os

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altum_v1.utils import load_agent_configs, create_tool_instances, initialize_agents, save_messages, load_previous_summaries, format_task_prompt

load_dotenv()

OVERALL_TASK_TEXT="""Your goal is to build an epigenetic clock capable of achieving near-SOTA performance in the task
of predicting chronological age from DNA methylation data. You have been provided with two files containing 
approximately 13k DNAm samples from patients of different ages with age, tissue type, and sex labels:

- `betas.arrow`: a feather file containing the beta values for each sample. The rows are the sample IDs and the columns are the probes IDs.
- `metadata.arrow`: a feather file containing the metadata for each sample. The rows are the sample IDs and the columns are the metadata.

The data comes from the 450k array. 

To achieve success, you will need to obtain at least 0.9 Pearson correlation between your predicted ages and real ages
in a held-out test set which I will not be providing you. The current best correlation achieved is 0.93, so this is 
a feasible goal. 
"""


async def run_understanding_task():
    """Run the task to understand the problem."""
    # Load previous summaries
    previous_summaries = await load_previous_summaries(1)
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['senior_advisor', 'principal_scientist']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)
    
    # Create agent group for the understanding task
    text_termination = TextMentionTermination("TERMINATE")
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination
    )
    
    # Define the initial task
    initial_task = f"""
    # CURRENT WORKFLOW STEP: Step 1 - Understanding the task

    Your team has been provided with the following task:

    > {OVERALL_TASK_TEXT}

    Please address the following questions:
    1. What is the purpose of this task? How does it relate to the overall goals of the aging research field?
    2. What prior studies have been done on this type of task? What are the main approaches and methods used?
    3. What are the main challenges and limitations of the prior studies?
    4. What are the most promising approaches and methods for this task?
    5. What is the nature of the data? What normalizations and transformations should be considered?
    6. How should the data be explored or visualized? What are some typical QC approaches for it?

    The next step in the workflow is Exploratory Data Analysis. So make sure to address these questions and ANY OTHERS
    that would be useful for EDA in the next step.

    Remember to work as a team, building on each other's insights. The Principal Scientist will summarize the discussion 
    and identify next steps when sufficient understanding has been reached.
    """
    
    # Format the task with previous context
    formatted_task = await format_task_prompt(initial_task, previous_summaries)
    
    # Run the agent group on the task
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description
    await save_messages(1, result.messages, result.messages[-1].dump()["content"], initial_task)
    
    return result

async def main():

    await run_understanding_task()

if __name__ == "__main__":
    asyncio.run(main())


