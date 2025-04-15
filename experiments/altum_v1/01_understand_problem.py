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
    prompt_for_workflow_action,
    setup_task_environment,
    resume_from_checkpoint,
    get_task_text,
    get_agent_token
)

load_dotenv()

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3
MODEL_TRAINING_STAGE = 4


async def run_understanding_task():
    """Run the task to understand the problem."""
    # Initialize workflow state for understanding stage
    stage = UNDERSTANDING_STAGE
    subtask = 1  # Only one subtask in understanding stage
    iteration = 1  # Only runs once
    
    # Get or create task environment
    task_env = setup_task_environment(stage, subtask, is_restart=True)
    
    # Save checkpoint at the start
    save_workflow_checkpoint(stage, subtask, iteration, "Understanding Start")
    
    # Update workflow state
    update_workflow_state(stage, subtask, iteration)
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['senior_advisor', 'principal_scientist']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)
    
    # Get the principal scientist's termination token
    principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")
    
    # Create agent group for the understanding task
    text_termination = TextMentionTermination(principal_scientist_termination_token)
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination,
        max_turns=5  # Allow enough turns for thorough discussion
    )
    
    # Get the task text from the config file
    task_text = get_task_text('understanding', 'task_1')
    
    # Format the task with structured format
    formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
    
    # Run the agent group on the task
    result = await Console(task_group.run_stream(task=formatted_task, output_stats=True))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
    
    # Save checkpoint after completion
    save_workflow_checkpoint(stage, subtask, iteration, "Understanding Complete")
    
    # Update workflow state to mark understanding stage as completed
    mark_stage_completed(UNDERSTANDING_STAGE)
    
    # Create checkpoint for the start of the next stage
    save_workflow_checkpoint(EDA_STAGE, label="Ready for EDA")
    
    return result

async def main(args=None):
    """Run the understanding phase workflow with checkpoint/resume capability."""
    
    # Default values
    restart_stage = UNDERSTANDING_STAGE
    is_restart = False
    
    if not args:
        # If running from command line, get workflow options
        action = await prompt_for_workflow_action(UNDERSTANDING_STAGE)
        
        # Handle different workflow actions
        if action["action"] == "restart":
            print(f"Restarting from stage {action['stage']}...")
            # Clear state for this stage and future stages
            await clear_workflow_state(action["stage"])
            restart_stage = action["stage"]
            is_restart = True
        elif action["action"] == "resume":
            print(f"Resuming from checkpoint {action['checkpoint_id']}...")
            # Restore state from checkpoint
            state = await resume_from_checkpoint(action["checkpoint_id"])
            # Resume operation would require more specific handling
            # For now, we just restart the task since it's a single subtask
            restart_stage = state["current_stage"]
        else:  # New workflow
            print("Starting new understanding workflow...")
            # Clear any existing state
            await clear_workflow_state(UNDERSTANDING_STAGE)
            restart_stage = UNDERSTANDING_STAGE
            is_restart = True
    else:
        # If called programmatically with args
        restart_stage = args.get("restart_stage", UNDERSTANDING_STAGE)
        is_restart = args.get("clear_state", False)
        if is_restart:
            await clear_workflow_state(restart_stage)
    
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


