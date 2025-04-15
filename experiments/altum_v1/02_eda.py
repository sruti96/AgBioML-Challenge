# This team will have two components:
# A. A team of agents that discuss the EDA and generate a specification for the engineer
# B. An engineer and code execution agent that writes the code to carry out the specification and returns the results
# The workflow is:
# Subtask 1: The team A agents discuss the EDA and generate a specification for the engineer
# Subtask 2: The team B agents write and execute the code to carry out the specification and the critic reviews the results.
# Subtask 2 continues with revisions until the critic approves the implementation.

import asyncio
import sys
import argparse
import glob
import os
import shutil
import traceback
import time

from dotenv import load_dotenv
import os

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

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
    is_stage_completed,
    get_maximum_iteration,
    clear_workflow_state,
    prompt_for_workflow_action,
    setup_task_environment,
    resume_from_checkpoint,
    save_structured_summary,
    get_task_text,
    get_agent_token
)
from altum_v1.agents import EngineerSociety

load_dotenv()

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3
MODEL_TRAINING_STAGE = 4

# Maximum number of retries for each subtask
MAX_RETRIES = 3

# Clean up temporary code files
def cleanup_temp_files(directory="."):
    """Remove temporary code files created during execution.
    
    Args:
        directory: Directory to clean (defaults to current directory)
    """
    # Pattern for temporary code files
    patterns = ["tmp_code_*", "*.pyc", "__pycache__"]
    
    total_removed = 0
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(directory, pattern)):
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Removed: {file_path}")
                    total_removed += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
                    total_removed += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    print(f"Cleanup complete. Removed {total_removed} temporary files/directories.")
    return total_removed

async def run_subtask_1(task_env=None):
    """Run the first subtask: Team discussion to create EDA specification."""
    # Get the workflow state
    state = get_workflow_state()
    stage = EDA_STAGE
    subtask = 1
    iteration = 1  # Subtask 1 only runs once
    
    # Update workflow state to indicate we're in EDA stage, subtask 1
    update_workflow_state(stage, subtask, iteration)
    
    # Get or create task environment if not provided
    if task_env is None:
        task_env = setup_task_environment(stage, subtask, is_restart=True)
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['principal_scientist', 'bioinformatics_expert', 'ml_expert']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    # Get the principal scientist's termination token
    principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")

    # Create agent group for the discussion task
    text_termination = TextMentionTermination(principal_scientist_termination_token)
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination,
        max_turns=5  # Allow enough turns for thorough discussion
    )
    
    # Get the task text from the config file
    task_text = get_task_text('eda', 'subtask_1')
    
    # Format the task with previous context
    formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
    
    result = await Console(task_group.run_stream(task=formatted_task, output_stats=True))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
    
    return result, task_env

async def run_subtask_2(iteration=1, task_env=None, retry_count=0):
    """Run the second subtask: Engineer implementing the EDA specification.
    
    Args:
        iteration: The iteration number for this subtask
        task_env: Optional task environment including workdir
        retry_count: Current retry attempt (0 = first attempt)
        
    Returns:
        Result object or None if all retries failed
    """
    # Update workflow state
    stage = EDA_STAGE
    subtask = 2
    update_workflow_state(stage, subtask, iteration)
    
    # Get or create task environment if not provided
    if task_env is None:
        # This is not a resume, so we might need to clean the directory
        is_restart = iteration == 1
        task_env = setup_task_environment(stage, subtask, is_restart=is_restart)
    
    try:
        # Clean up any temp files from previous runs
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {MAX_RETRIES}...")
            cleanup_temp_files()
            # Short pause to ensure cleanup is complete
            time.sleep(1)
        
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        
        # Initialize engineer team
        engineer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['engineer'], tools=available_tools)['engineer']
        engineer_termination_token = get_agent_token(agent_configs, "engineer")
        # Add code executor to the engineer team
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=task_env["workdir"],  # Use main working directory for Docker access
            timeout=600
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
        engineer_team = RoundRobinGroupChat(
            participants=[engineer_agent, code_executor_agent],
            termination_condition=TextMentionTermination(engineer_termination_token),
            max_turns=50
        )
        
        # Initialize critic team
        critic_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['data_science_critic'], tools=available_tools)['data_science_critic']
        critic_termination_token = get_agent_token(agent_configs, "data_science_critic")
        critic_team = RoundRobinGroupChat(
            participants=[critic_agent],
            termination_condition=TextMentionTermination(critic_termination_token)
        )
        
        # Initialize summarizer agent
        summarizer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['summarizer'], tools=available_tools)['summarizer']
        
        # Choose the appropriate task text based on iteration
        if iteration == 1:
            task_text = get_task_text('eda', 'subtask_2')
        else:
            # Format the revision text with the current iteration number
            task_text = get_task_text('eda', 'subtask_2_revision', iteration=iteration)
        
        # Append directory information to the task text
        task_text += f"\n\nIMPORTANT FILE ORGANIZATION INSTRUCTIONS:"
        task_text += f"\n- Your code runs in the main project directory where data files are located"
        task_text += f"\n- SAVE ALL OUTPUT FILES to the '{task_env['output_dir']}' directory"
        task_text += f"\n- This includes plots, intermediate data files, and any other outputs"
        task_text += f"\n- Example: `plt.savefig('{task_env['output_dir']}/my_plot.png')`"
        
        # Add information about data file locations
        data_files_text = "\n\nDATA FILE INFORMATION:"
        for filename, file_info in task_env['data_files'].items():
            if file_info["location"] == "current_dir":
                data_files_text += f"\n- File '{filename}' is in the current working directory"
            else:
                data_files_text += f"\n- File '{filename}' status: {file_info['status']}"
        
        # Add token management warnings for retry attempts
        if retry_count > 0:
            data_files_text += f"\n\n⚠️ CRITICAL WARNING: Previous attempt failed due to token overflow ⚠️"
            data_files_text += f"\n- Be EXTREMELY careful with output sizes"
            data_files_text += f"\n- NEVER print large arrays or full dataframes"
            data_files_text += f"\n- Use .head(), .describe(), and sampling aggressively"
            data_files_text += f"\n- Consider saving outputs to files instead of printing"
        
        task_text += data_files_text
        
        # Format the task with previous context, including the current iteration
        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        # Create the task message
        task_message = TextMessage(
            content=formatted_task,
            source="User"
        )
        
        # Create the EngineerSociety that manages the interaction between teams
        engineer_society = EngineerSociety(
            name="eda_society",
            engineer_team=engineer_team,
            critic_team=critic_team,
            critic_approve_token=get_agent_token(agent_configs, "data_science_critic", "approval_token"),
            engineer_terminate_token=get_agent_token(agent_configs, "engineer"),
            critic_terminate_token=get_agent_token(agent_configs, "data_science_critic"),
            critic_revise_token=get_agent_token(agent_configs, "data_science_critic", "revision_token"),
            summarizer_agent=summarizer_agent,
            original_task=task_text,
            output_dir=task_env['output_dir']
        )
        
        # Run the engineer society with the formatted task
        print(f"Starting EngineerSociety execution for EDA (iteration {iteration})...")
        result = await engineer_society.on_messages([task_message], CancellationToken())
        
        # Extract messages for saving
        engineer_messages = [task_message]  # Start with the task message
        if result.inner_messages:
            engineer_messages.extend(result.inner_messages)
        if result.chat_message:
            engineer_messages.append(result.chat_message)
        
        # Get the content of the result for the summary
        summary_content = result.chat_message.content if result.chat_message else "No result"
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, engineer_messages, 
                                     summary_content, task_text)
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
        # Save the summarized report for future reference
        if result.chat_message and isinstance(result.chat_message, TextMessage):
            summary_file_path = os.path.join(task_env['output_dir'], f"implementation_summary_iteration_{iteration}.txt")
            with open(summary_file_path, "w") as f:
                f.write(result.chat_message.content)
            print(f"Saved implementation summary to {summary_file_path}")
            
            # Also save the implementation summary to the structured summaries
            await save_structured_summary(stage, subtask, iteration, result.chat_message.content, task_text)
        
        return result
        
    except Exception as e:
        print(f"Error in subtask 2 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        
        # If we haven't exhausted retries, try again
        if retry_count < MAX_RETRIES - 1:
            print(f"Retrying subtask 2 (iteration {iteration})...")
            return await run_subtask_2(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries ({MAX_RETRIES}) exceeded for subtask 2. Giving up.")
            return None

async def main(args=None):
    """Run the complete EDA workflow."""
    
    # Default values
    restart_stage = EDA_STAGE
    is_restart = False
    resume_checkpoint = None
    task_env = None
    
    if not args:
        # If running from command line, get workflow options
        action = await prompt_for_workflow_action(EDA_STAGE)
        
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
            # Get current stage from restored state
            restart_stage = state["current_stage"]
            resume_checkpoint = action["checkpoint_id"]
            
            # Get the latest subtask and iteration
            if "iterations" in state:
                stage_key = f"stage{restart_stage}"
                if stage_key in state["iterations"]:
                    subtasks = state["iterations"][stage_key]
                    if subtasks:
                        # Find highest subtask and its iteration
                        highest_subtask = max(int(s.replace("subtask", "")) for s in subtasks.keys())
                        subtask_key = f"subtask{highest_subtask}"
                        current_iteration = subtasks.get(subtask_key, 1)
                        
                        # Create task environment for resumed task (don't clean directory)
                        task_env = setup_task_environment(restart_stage, highest_subtask, is_restart=False)
                        
                        # Resume from this point
                        print(f"Resuming at Stage {restart_stage}, Subtask {highest_subtask}, Iteration {current_iteration}")
                        
                        if highest_subtask == 1:  # Already completed subtask 1
                            print("Subtask 1 already completed, skipping...")
                        elif highest_subtask == 2:  # In the middle of subtask 2
                            print(f"Resuming at subtask 2, iteration {current_iteration}...")
                            result2 = await run_subtask_2(current_iteration, task_env)
                            if not result2:
                                print(f"Subtask 2 (iteration {current_iteration}) failed to complete")
                                return
                            
                            # Mark stage as completed since the EngineerSociety completes when approved
                            print(f"EDA completed after iteration {current_iteration}")
                            mark_stage_completed(EDA_STAGE)
                            save_workflow_checkpoint(DATA_SPLIT_STAGE, label="Ready for Data Splitting")
                            return
                        
                        # If we get here, we're done with the resume-specific logic
                        return
        else:  # New workflow
            print("Starting new EDA workflow...")
            # Set workflow state to EDA stage
            update_workflow_state(EDA_STAGE)
            restart_stage = EDA_STAGE
            is_restart = True
    else:
        # If called programmatically with args
        restart_stage = args.get("restart_stage", EDA_STAGE)
        is_restart = args.get("clear_state", False)
        if is_restart:
            await clear_workflow_state(restart_stage)
    
    # Clean up any temporary files from previous runs
    cleanup_temp_files()
    
    # If we haven't already created a task environment during resume
    if task_env is None:
        # Set up the task environment
        task_env = setup_task_environment(restart_stage, is_restart=is_restart)
    
    # Save a checkpoint at the start
    save_workflow_checkpoint(EDA_STAGE, label="EDA Start")
    
    # Start the workflow from the appropriate point
    if restart_stage == EDA_STAGE:
        # Check if understanding stage is completed
        if not is_stage_completed(UNDERSTANDING_STAGE):
            print("Warning: Understanding stage (Stage 1) has not been completed.")
            proceed = input("Do you want to proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Exiting. Please run Stage 1 first.")
                return
    
        # Run subtask 1: Team discussion to create EDA specification
        result1, task_env = await run_subtask_1(task_env)
        if not result1:
            print("Subtask 1 failed to complete")
            return
        
        # Save checkpoint after subtask 1
        save_workflow_checkpoint(EDA_STAGE, 1, 1, "EDA Specification Completed")
    
    # Run subtask 2: Engineer implementing the EDA
    iteration = 1
    result2 = await run_subtask_2(iteration, task_env)
    if not result2:
        print("Subtask 2 failed to complete after multiple retries")
        return
    
    # Save checkpoint after subtask 2
    save_workflow_checkpoint(EDA_STAGE, 2, iteration, f"EDA Implementation (Iteration {iteration})")
    
    # Mark stage as completed (EngineerSociety handles the approval)
    print(f"EDA completed after iteration {iteration}")
    mark_stage_completed(EDA_STAGE)
    save_workflow_checkpoint(DATA_SPLIT_STAGE, label="Ready for Data Splitting")
    
    # Final cleanup of temporary files
    cleanup_temp_files()

if __name__ == "__main__":
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Run the EDA workflow with checkpoint/resume options")
    parser.add_argument("--restart", action="store_true", help="Restart from beginning of EDA stage")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--force", action="store_true", help="Force restart without prompting")
    args_parsed = parser.parse_args()
    
    if args_parsed.restart and args_parsed.resume:
        print("Error: Cannot specify both --restart and --resume")
        sys.exit(1)
    
    if args_parsed.restart:
        # Restart from beginning of EDA
        asyncio.run(clear_workflow_state(EDA_STAGE))
        asyncio.run(main({"restart_stage": EDA_STAGE, "clear_state": True}))
    elif args_parsed.resume:
        # Resume from latest checkpoint (interactive)
        asyncio.run(main())
    else:
        # Interactive mode
        asyncio.run(main())





