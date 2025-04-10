import asyncio
import yaml
import argparse

from dotenv import load_dotenv
import os

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altum_v1.utils import (
    load_agent_configs, 
    create_tool_instances, 
    initialize_agents, 
    format_structured_task_prompt,
    save_messages_structured,
    get_workflow_state,
    update_workflow_state,
    save_workflow_checkpoint,
    mark_stage_completed,
    clear_workflow_state,
    prompt_for_workflow_action
)

load_dotenv()

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
EXPERIMENTAL_DESIGN_STAGE = 3
MODEL_TRAINING_STAGE = 4

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
    # Initialize workflow state for understanding stage
    stage = UNDERSTANDING_STAGE
    subtask = 1  # Only one subtask in understanding stage
    iteration = 1  # Only runs once
    
    # Save checkpoint at the start
    save_workflow_checkpoint(stage, subtask, iteration, "Understanding Start")
    
    # Update workflow state
    update_workflow_state(stage, subtask, iteration)
    
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
    
    # Format the task with structured format
    formatted_task = await format_structured_task_prompt(stage, subtask, initial_task, iteration)
    
    # Run the agent group on the task
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], initial_task)
    
    # Save checkpoint after completion
    save_workflow_checkpoint(stage, subtask, iteration, "Understanding Complete")
    
    # Update workflow state to mark understanding stage as completed
    mark_stage_completed(UNDERSTANDING_STAGE)
    
    # Create checkpoint for the start of the next stage
    save_workflow_checkpoint(EDA_STAGE, label="Ready for EDA")
    
    return result

async def main(args=None):
    """Run the understanding phase workflow with checkpoint/resume capability."""
    
    if not args:
        # If running from command line, get workflow options
        action = await prompt_for_workflow_action(UNDERSTANDING_STAGE)
        
        # Handle different workflow actions
        if action["action"] == "restart":
            print(f"Restarting from stage {action['stage']}...")
            # Clear state for this stage and future stages
            await clear_workflow_state(action["stage"])
        elif action["action"] == "resume":
            print(f"Resuming from checkpoint {action['checkpoint_id']}...")
            # Restore state from checkpoint
            await resume_from_checkpoint(action["checkpoint_id"])
            # Resume operation would require more specific handling
            # For now, we just restart the task since it's a single subtask
        else:  # New workflow
            print("Starting new understanding workflow...")
            # Clear any existing state
            await clear_workflow_state(UNDERSTANDING_STAGE)
    else:
        # If called programmatically with args
        if args.get("clear_state", False):
            await clear_workflow_state(UNDERSTANDING_STAGE)
    
    # Run the understanding task
    await run_understanding_task()
    
    print("Understanding phase completed. Ready to proceed to EDA phase.")

if __name__ == "__main__":
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Run the Understanding workflow with checkpoint/resume options")
    parser.add_argument("--restart", action="store_true", help="Restart from beginning")
    parser.add_argument("--force", action="store_true", help="Force restart without prompting")
    args_parsed = parser.parse_args()
    
    if args_parsed.restart:
        # Restart from beginning
        asyncio.run(main({"clear_state": True}))
    else:
        # Interactive mode
        asyncio.run(main())


