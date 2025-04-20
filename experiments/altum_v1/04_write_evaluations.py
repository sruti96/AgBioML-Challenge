"""
This module handles the model evaluation stage of the workflow.
It uses agents to collaborate on defining an evaluation strategy and implementing an evaluation script.
"""

import argparse
import asyncio
import os
import sys
import traceback
import time
import glob
import shutil

from dotenv import load_dotenv

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent
from docker.types import DeviceRequest

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
    clear_workflow_state,
    prompt_for_workflow_action,
    setup_task_environment,
    resume_from_checkpoint,
    get_task_text,
    get_agent_token,
    save_structured_summary
)
from altum_v1.agents import EngineerSociety

load_dotenv()

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3
MODEL_EVALUATION_STAGE = 4 # Evaluation script writing is Stage 4
MODEL_BUILDING_STAGE = 5   # Model building will be Stage 5.

# Maximum number of retries for each subtask
MAX_RETRIES = 3

# Clean up temporary code files
def cleanup_temp_files(directory="."):
    """Remove temporary code files created during execution."""
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
    if total_removed > 0:
        print(f"Cleanup complete. Removed {total_removed} temporary files/directories.")
    return total_removed

async def run_subtask_1(task_env=None):
    """Run the first subtask: Team discussion to define evaluation strategy."""
    stage = MODEL_EVALUATION_STAGE
    subtask = 1
    iteration = 1  # Subtask 1 only runs once
    
    update_workflow_state(stage, subtask, iteration)
    
    if task_env is None:
        task_env = setup_task_environment(stage, subtask, is_restart=True)
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    # Define Team A for evaluation planning
    which_agents = ['principal_scientist', 'ml_expert', 'bioinformatics_expert'] # Added bio expert
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")

    text_termination = TextMentionTermination(principal_scientist_termination_token)
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination,
        max_turns=5
    )
    
    task_text = get_task_text('create_evaluation', 'subtask_1') # Use create_evaluation category
    formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
    
    print(f"Starting Evaluation Strategy Discussion (Stage {stage}, Subtask {subtask})...")
    result = await Console(task_group.run_stream(task=formatted_task), output_stats=True)
    
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
    
    return result, task_env

async def run_subtask_2(iteration=1, task_env=None, retry_count=0):
    """Run the second subtask: Engineer implementing the evaluation script."""
    stage = MODEL_EVALUATION_STAGE
    subtask = 2
    update_workflow_state(stage, subtask, iteration)
    
    if task_env is None:
        is_restart = iteration == 1
        task_env = setup_task_environment(stage, subtask, is_restart=is_restart)
    
    try:
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {MAX_RETRIES}...")
            cleanup_temp_files(task_env['output_dir']) # Clean specific output dir
            time.sleep(1)
        
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        
        # Initialize engineer team
        engineer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['engineer'], tools=available_tools)['engineer']
        engineer_termination_token = get_agent_token(agent_configs, "engineer")
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=task_env["workdir"],
            timeout=300,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
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
        
        if iteration == 1:
            task_text = get_task_text('create_evaluation', 'subtask_2') # Use create_evaluation category
        else:
            # Revision task description is not defined in tasks.yaml yet, adjust if needed
            # task_text = get_task_text('create_evaluation', 'subtask_2_revision', iteration=iteration)
             # For now, let's assume the critic provides enough context for revision
             task_text = get_task_text('create_evaluation', 'subtask_2') + "\n\n# REVISION INSTRUCTIONS:\nPlease address the feedback provided by the critic in the previous messages."
        
        # Add instructions specific to this task (e.g., dummy data paths)
        task_text += f"""\n\nIMPORTANT CONTEXT:
        - Save the evaluation script as `evaluation_script.py` in the output directory: {task_env['output_dir']}
        - Save any generated plots or results tables to the same directory: {task_env['output_dir']}
        - For testing purposes, assume dummy input files named `dummy_true.csv` and `dummy_pred.csv` exist in the workdir ({task_env['workdir']}). Your test code should create these briefly if needed.
        
        """

        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        task_message = TextMessage(content=formatted_task, source="User")
        
        engineer_society = EngineerSociety(
            name="evaluation_script_society",
            engineer_team=engineer_team,
            critic_team=critic_team,
            critic_approve_token=get_agent_token(agent_configs, "data_science_critic", "approval_token"),
            engineer_terminate_token=engineer_termination_token, # Use the token directly
            critic_terminate_token=critic_termination_token,
            critic_revise_token=get_agent_token(agent_configs, "data_science_critic", "revision_token"),
            summarizer_agent=summarizer_agent,
            original_task=task_text, # Pass the base task text for summarization context
            output_dir=task_env['output_dir']
        )
        
        print(f"Starting EngineerSociety execution for Evaluation Script (Stage {stage}, Subtask {subtask}, Iteration {iteration})...")
        result = await engineer_society.on_messages([task_message], CancellationToken())
        
        engineer_messages = [task_message] 
        if result.inner_messages:
            engineer_messages.extend(result.inner_messages)
        if result.chat_message:
            engineer_messages.append(result.chat_message)
        
        summary_content = result.chat_message.content if result.chat_message else "No final summary provided by EngineerSociety."
        
        await save_messages_structured(stage, subtask, iteration, engineer_messages, summary_content, task_text)
        
        # Save the summary to a text file as well
        if result.chat_message and isinstance(result.chat_message, TextMessage):
            summary_file_path = os.path.join(task_env['output_dir'], f"evaluation_summary_iteration_{iteration}.txt")
            with open(summary_file_path, "w") as f:
                f.write(summary_content)
            print(f"Saved evaluation summary to {summary_file_path}")
            # Ensure it's saved to structured memory too
            await save_structured_summary(stage, subtask, iteration, summary_content, task_text)

        cleanup_temp_files(task_env['output_dir'])
        cleanup_temp_files(task_env['workdir'])
        
        return result
        
    except Exception as e:
        print(f"Error in evaluation subtask 2 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        if retry_count < MAX_RETRIES - 1:
            print(f"Retrying evaluation subtask 2 (iteration {iteration})...")
            return await run_subtask_2(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries ({MAX_RETRIES}) exceeded for evaluation subtask 2. Giving up.")
            return None

async def main(args=None):
    """Run the complete model evaluation script writing workflow."""
    restart_stage = MODEL_EVALUATION_STAGE
    is_restart = False
    resume_checkpoint = None
    task_env = None
    
    if not args:
        action = await prompt_for_workflow_action(MODEL_EVALUATION_STAGE)
        if action["action"] == "restart":
            print(f"Restarting from stage {action['stage']}...")
            await clear_workflow_state(action["stage"])
            restart_stage = action["stage"]
            is_restart = True
        elif action["action"] == "resume":
            print(f"Resuming from checkpoint {action['checkpoint_id']}...")
            state = await resume_from_checkpoint(action["checkpoint_id"])
            restart_stage = state["current_stage"]
            resume_checkpoint = action["checkpoint_id"]
            # Get the latest subtask and iteration for resume
            if "iterations" in state:
                stage_key = f"stage{restart_stage}"
                if stage_key in state["iterations"]:
                    subtasks = state["iterations"][stage_key]
                    if subtasks:
                        highest_subtask = max(int(s.replace("subtask", "")) for s in subtasks.keys())
                        subtask_key = f"subtask{highest_subtask}"
                        current_iteration = subtasks.get(subtask_key, 1)
                        task_env = setup_task_environment(restart_stage, highest_subtask, is_restart=False)
                        print(f"Resuming at Stage {restart_stage}, Subtask {highest_subtask}, Iteration {current_iteration}")
                        if highest_subtask == 1:
                            print("Subtask 1 (Strategy) already completed, skipping...")
                        elif highest_subtask == 2:
                            print(f"Resuming at subtask 2 (Implementation), iteration {current_iteration}...")
                            result2 = await run_subtask_2(current_iteration, task_env)
                            if not result2:
                                print(f"Subtask 2 (iteration {current_iteration}) failed to complete upon resume.")
                                return
                            print(f"Model Evaluation Script stage completed after iteration {current_iteration}.")
                            mark_stage_completed(MODEL_EVALUATION_STAGE)
                            save_workflow_checkpoint(MODEL_BUILDING_STAGE, label="Ready for Model Building")
                            print("Workflow proceeding to Model Building Stage.")
                            return
                        return # End resume logic
        else: # New workflow
            print("Starting new Model Evaluation Script workflow...")
            await clear_workflow_state(MODEL_EVALUATION_STAGE)
            restart_stage = MODEL_EVALUATION_STAGE
            is_restart = True
    else: # Programmatic start
        restart_stage = args.get("restart_stage", MODEL_EVALUATION_STAGE)
        is_restart = args.get("clear_state", False)
        if is_restart:
            await clear_workflow_state(restart_stage)
    
    cleanup_temp_files() # General cleanup before start
    
    if task_env is None:
        task_env = setup_task_environment(restart_stage, is_restart=is_restart)
    
    save_workflow_checkpoint(MODEL_EVALUATION_STAGE, label="Model Evaluation Script Start")
    
    # Only run stages if starting from or before MODEL_EVALUATION_STAGE
    if restart_stage <= MODEL_EVALUATION_STAGE:
        # Check if Data Splitting (Stage 3) is completed
        if not is_stage_completed(DATA_SPLIT_STAGE):
            print(f"Warning: Data Splitting stage ({DATA_SPLIT_STAGE}) has not been completed.")
            proceed = input("Do you want to proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print(f"Exiting. Please run Stage {DATA_SPLIT_STAGE} first.")
                return
                
        # Run subtask 1: Define evaluation strategy
        # Check if subtask 1 for this stage already exists in state
        state = get_workflow_state()
        subtask1_completed = False
        if f"stage{MODEL_EVALUATION_STAGE}" in state.get("iterations", {}):
             if "subtask1" in state["iterations"][f"stage{MODEL_EVALUATION_STAGE}"]:
                 subtask1_completed = True

        if not subtask1_completed: 
             print("\n--- Running Subtask 1: Define Evaluation Strategy ---")
             result1, task_env = await run_subtask_1(task_env)
             if not result1:
                 print("Subtask 1 (Evaluation Strategy) failed to complete")
                 return
             save_workflow_checkpoint(MODEL_EVALUATION_STAGE, 1, 1, "Evaluation Strategy Defined")
        else:
             print("\n--- Skipping Subtask 1: Define Evaluation Strategy (already completed) ---")

    # Determine starting iteration for subtask 2 based on latest state
    start_iteration = 1
    state = get_workflow_state() # Re-fetch state in case subtask 1 just ran
    stage_key = f"stage{MODEL_EVALUATION_STAGE}"
    subtask_key = "subtask2"
    if stage_key in state.get("iterations", {}) and subtask_key in state["iterations"][stage_key]:
        # If resuming or restarting after subtask 1 completed, start from the recorded iteration
        start_iteration = state["iterations"][stage_key][subtask_key] 
    
    # Run subtask 2: Implement evaluation script
    print(f"\n--- Running Subtask 2: Implement Evaluation Script (Starting Iteration {start_iteration}) ---")
    result2 = await run_subtask_2(start_iteration, task_env)
    if not result2:
        print(f"Subtask 2 (Implement Evaluation Script) failed to complete after multiple retries (starting iteration {start_iteration})")
        return
    
    # Determine the final iteration based on where run_subtask_2 finished
    final_state = get_workflow_state()
    final_iteration = final_state.get("iterations", {}).get(stage_key, {}).get(subtask_key, start_iteration)
    
    save_workflow_checkpoint(MODEL_EVALUATION_STAGE, 2, final_iteration, f"Evaluation Script Implementation (Iteration {final_iteration})")
    
    # Mark stage as completed (EngineerSociety handles the approval)
    print(f"\nModel Evaluation Script stage completed after iteration {final_iteration}.")
    mark_stage_completed(MODEL_EVALUATION_STAGE)
    save_workflow_checkpoint(MODEL_BUILDING_STAGE, label="Ready for Model Building")
    print("Workflow proceeding to Model Building Stage.")
    
    cleanup_temp_files() # Final cleanup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model Evaluation Script workflow stage.")
    parser.add_argument("--restart", action="store_true", help="Restart from beginning of this stage")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint for this stage")
    args_parsed = parser.parse_args()
    
    if args_parsed.restart and args_parsed.resume:
        print("Error: Cannot specify both --restart and --resume")
        sys.exit(1)
    
    if args_parsed.restart:
        asyncio.run(main({"restart_stage": MODEL_EVALUATION_STAGE, "clear_state": True}))
    elif args_parsed.resume:
        asyncio.run(main())
    else:
        # Default behavior: prompt user
        asyncio.run(main())
