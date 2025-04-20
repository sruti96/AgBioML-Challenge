"""
This module handles the iterative model improvement stage of the workflow.
Team A reviews results and plans changes, then Team B implements and evaluates them.
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
MODEL_EVALUATION_STAGE = 4 
MODEL_BUILDING_STAGE = 5
TRAIN_EVALUATE_STAGE = 6
REVIEW_ITERATE_STAGE = 7 
# Define next stage - could loop back or go to a final summary
# NEXT_STAGE = MODEL_BUILDING_STAGE # Example if looping back to rebuild
FINAL_REPORTING_STAGE = 8 # Example: If this is the last iteration cycle

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

async def run_subtask_1(task_env=None, iteration=1):
    """Run Subtask 1: Team A reviews results and plans the next iteration."""
    stage = REVIEW_ITERATE_STAGE
    subtask = 1
    
    update_workflow_state(stage, subtask, iteration)
    
    if task_env is None:
        task_env = setup_task_environment(stage, subtask, is_restart=True)
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['principal_scientist', 'ml_expert', 'bioinformatics_expert'] 
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")
    text_termination = TextMentionTermination(principal_scientist_termination_token)
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination,
        max_turns=15 
    )
    
    task_text = get_task_text('review_iterate', 'subtask_1')
    
    # Add context about previous stage output locations
    task_text += """\n\nADDITIONAL CONTEXT - PREVIOUS RESULTS:
    - Review evaluation results from Stage 6 (task_6_workdir)
    - Review the results of the previous iteration of the current task you are working on (if there is one)
    """
    # TODO: Add specific file paths dynamically if needed

    formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
    
    print(f"Starting Review & Iteration Planning (Stage {stage}, Subtask {subtask})...")
    result = await Console(task_group.run_stream(task=formatted_task), output_stats=True)
    
    summary_content = result.messages[-1].dump()["content"]
    await save_messages_structured(stage, subtask, iteration, result.messages, summary_content, task_text)
    print(f"\n--- Stage {stage}, Subtask 1 Completed: Specification for next iteration generated ---")
    
    return result, task_env

async def run_subtask_2(iteration=1, task_env=None, retry_count=0):
    """Run Subtask 2: Engineer implements the iteration plan and evaluates."""
    stage = REVIEW_ITERATE_STAGE
    subtask = 2
    timeout = 1800
    # Iteration number here reflects the *overall* model improvement loop count, passed from main
    update_workflow_state(stage, subtask, iteration) 
    
    if task_env is None:
        # Use the *same* task_env as subtask 1 for this stage, but don't clean if retrying
        is_restart = retry_count == 0 
        task_env = setup_task_environment(stage, subtask, is_restart=is_restart, workdir_suffix=f"iteration_{iteration}")
    
    try:
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {MAX_RETRIES}...")
            cleanup_temp_files(task_env['output_dir']) 
            cleanup_temp_files(task_env['workdir']) 
            time.sleep(1)
        
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        
        engineer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['engineer'], tools=available_tools)['engineer']
        engineer_termination_token = get_agent_token(agent_configs, "engineer")
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest', 
            work_dir=task_env["workdir"],
            timeout=timeout,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor, character_limit=10_000)
        engineer_team = RoundRobinGroupChat(
            participants=[engineer_agent, code_executor_agent],
            termination_condition=TextMentionTermination(engineer_termination_token),
            max_turns=200 
        )
        
        critic_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['data_science_critic'], tools=available_tools)['data_science_critic']
        critic_termination_token = get_agent_token(agent_configs, "data_science_critic")
        critic_team = RoundRobinGroupChat(
            participants=[critic_agent],
            termination_condition=TextMentionTermination(critic_termination_token)
        )
        
        summarizer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['summarizer'], tools=available_tools)['summarizer']
        
        # Subtask 2 always uses the 'review_iterate.subtask_2' description
        # The iteration number is passed for context within the prompt itself if needed
        task_text = get_task_text('review_iterate', 'subtask_2', iteration=iteration) 
        
        # Format with output/work directories
        # Need to reference previous stage directories correctly
        # Assuming task_X_workdir convention, get paths for stages 3, 4, 5, 6
        # This might require a helper function in utils or explicit passing
        # For now, using placeholder names - MUST BE FIXED for actual execution
        # data_dir = "task_3_workdir" # Stage 3
        # eval_script_dir = "task_4_workdir" # Stage 4
        # model_script_dir = "task_5_workdir" # Stage 5
        # prev_model_dir = "task_6_workdir" # Stage 6
        
        # Simplification: Assume engineer copies files using relative paths from parent
        task_text += f"""\n\nREMINDER:
- Your working directory: `{task_env['workdir']}`
- Your output directory for THIS iteration: `{task_env['output_dir']}`
- Data is in `task_3_workdir/`
- Evaluation script is in `task_4_workdir/`
- Previous training/prediction scripts are in `task_5_workdir/`
- Previous trained model is in `task_6_workdir/` (or previous iteration's output)
Use relative paths like `task_3_workdir/train.arrow` when copying or accessing files."""
        
        task_text += f"""\n\nREMINDER:
- The code executor has a STRICT {timeout} second timeout.
- Operations exceeding this limit will be terminated.
- You MUST plan your work in a way that avoids these timeouts.
- REMEMBER: you have access to 30 cores and 1 A10 GPU, along with 220GB of RAM.
"""

        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        task_message = TextMessage(content=formatted_task, source="User")
        
        engineer_society = EngineerSociety(
            name=f"review_iterate_society_iter_{iteration}",
            engineer_team=engineer_team,
            critic_team=critic_team,
            critic_approve_token=get_agent_token(agent_configs, "data_science_critic", "approval_token"),
            engineer_terminate_token=engineer_termination_token, 
            critic_terminate_token=critic_termination_token,
            critic_revise_token=get_agent_token(agent_configs, "data_science_critic", "revision_token"),
            summarizer_agent=summarizer_agent,
            original_task=task_text, 
            output_dir=task_env['output_dir']
        )
        
        print(f"Starting EngineerSociety execution for Review/Iterate (Stage {stage}, Subtask {subtask}, Iteration {iteration})...")
        result = await engineer_society.on_messages([task_message], CancellationToken())
        
        engineer_messages = [task_message] 
        if result.inner_messages:
            engineer_messages.extend(result.inner_messages)
        if result.chat_message:
            engineer_messages.append(result.chat_message)
        
        summary_content = result.chat_message.content if result.chat_message else "No final summary provided by EngineerSociety."
        
        await save_messages_structured(stage, subtask, iteration, engineer_messages, summary_content, task_text)
        
        if result.chat_message and isinstance(result.chat_message, TextMessage):
            summary_file_path = os.path.join(task_env['output_dir'], f"review_iterate_summary_iteration_{iteration}.txt")
            with open(summary_file_path, "w") as f:
                f.write(summary_content)
            print(f"Saved review/iterate summary to {summary_file_path}")
            await save_structured_summary(stage, subtask, iteration, summary_content, task_text)

        cleanup_temp_files(task_env['output_dir'])
        cleanup_temp_files(task_env['workdir'])
        
        return result
        
    except Exception as e:
        print(f"Error in review/iterate subtask 2 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        if retry_count < MAX_RETRIES - 1:
            print(f"Retrying review/iterate subtask 2 (iteration {iteration})...")
            return await run_subtask_2(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries ({MAX_RETRIES}) exceeded for review/iterate subtask 2. Giving up.")
            return None

async def main(args=None):
    """Run the complete review and iterate workflow stage."""
    restart_stage = REVIEW_ITERATE_STAGE
    is_restart = False
    task_env = None 
    # This stage might manage multiple iterations internally or externally
    # For now, assume one pass: plan then execute
    current_model_iteration = 1 # Example: track overall model iterations
    resume_checkpoint = None # Initialize as None
    if not args:
        action = await prompt_for_workflow_action(REVIEW_ITERATE_STAGE)
        if action["action"] == "restart":
            print(f"Restarting Stage {REVIEW_ITERATE_STAGE} from Subtask 1...")
            await clear_workflow_state(REVIEW_ITERATE_STAGE)
            restart_stage = REVIEW_ITERATE_STAGE
            is_restart = True
            current_model_iteration = args.get("iteration", 1) if args else 1 # Reset iteration count on restart
        elif action["action"] == "resume":
             print(f"Resuming from checkpoint {action['checkpoint_id']}...")
             state = await resume_from_checkpoint(action["checkpoint_id"])
             restart_stage = state["current_stage"]
             resume_checkpoint = action["checkpoint_id"] # Assign the checkpoint ID
             is_restart = True # Treat resume as a restart for this planning stage
             await clear_workflow_state(restart_stage) # Clear state to ensure it runs fresh
        else: # New workflow start for this stage
            print("Starting new Review & Iterate workflow...")
            await clear_workflow_state(REVIEW_ITERATE_STAGE)
            restart_stage = REVIEW_ITERATE_STAGE
            is_restart = True
            current_model_iteration = 1
    else: # Programmatic start
        restart_stage = args.get("restart_stage", REVIEW_ITERATE_STAGE)
        is_restart = args.get("clear_state", False)
        current_model_iteration = args.get("iteration", 1)
        if is_restart:
            await clear_workflow_state(restart_stage)
    
    cleanup_temp_files() 
    
    if task_env is None: # If not resuming mid-subtask 2
        task_env = setup_task_environment(restart_stage, 1, is_restart=is_restart, workdir_suffix=f"iteration_{current_model_iteration}") # Env for the whole stage iteration
    
    save_workflow_checkpoint(REVIEW_ITERATE_STAGE, label=f"Review & Iterate Start (Iteration {current_model_iteration})")
    
    subtask_to_run = 1
    if not is_restart and resume_checkpoint: # Check resume state
         state = get_workflow_state()
         stage_key = f"stage{REVIEW_ITERATE_STAGE}"
         if stage_key in state.get("iterations", {}):
             subtasks = state["iterations"][stage_key]
             highest_subtask = max(int(s.replace("subtask", "")) for s in subtasks.keys()) if subtasks else 0
             if highest_subtask == 1:
                 subtask_to_run = 2 # Start from subtask 2 if subtask 1 is done
             # If highest is 2, run_subtask_2 handles resume internally via iteration count

    # Run Subtask 1: Planning
    if subtask_to_run == 1:
        print(f"\n--- Running Subtask 1: Plan Iteration {current_model_iteration} ---")
        result1, task_env = await run_subtask_1(task_env, current_model_iteration) # Pass potentially existing task_env
        if not result1:
            print(f"Subtask 1 (Planning Iteration {current_model_iteration}) failed")
            return
        save_workflow_checkpoint(REVIEW_ITERATE_STAGE, 1, current_model_iteration, f"Iteration {current_model_iteration} Plan Defined")
        subtask_to_run = 2 # Proceed to subtask 2
    else:
        print(f"\n--- Skipping Subtask 1: Planning Iteration {current_model_iteration} (already completed or resuming later) ---")

    # Run Subtask 2: Execution
    if subtask_to_run == 2:
        print(f"\n--- Running Subtask 2: Execute Iteration {current_model_iteration} ---")
        result2 = await run_subtask_2(current_model_iteration, task_env) # Pass task_env from subtask 1
        if not result2:
            print(f"Subtask 2 (Execute Iteration {current_model_iteration}) failed")
            return
        # Checkpoint inside run_subtask_2 is more appropriate if it handles iterations internally
        # Here, we save checkpoint after the call completes for this stage iteration
        save_workflow_checkpoint(REVIEW_ITERATE_STAGE, 2, current_model_iteration, f"Iteration {current_model_iteration} Executed & Evaluated")

    # Mark stage as completed (conceptually, one loop is done)
    print(f"\nReview & Iterate Stage (Iteration {current_model_iteration}) completed.")
    mark_stage_completed(REVIEW_ITERATE_STAGE) # Mark completion for this pass
    
    # Decide if further iterations are needed (this requires external logic or user input)
    print("\nWorkflow Pause: Review the results of this iteration.")
    print(f"Outputs are in: {task_env['output_dir']}")
    print("To run another iteration, restart this stage potentially with --clear_state=False and --iteration=<next_iteration_number>")
    print(f"Or proceed to Stage {FINAL_REPORTING_STAGE} if performance is satisfactory.")
    save_workflow_checkpoint(FINAL_REPORTING_STAGE, label=f"Ready for Final Reporting after Iteration {current_model_iteration}")
    
    cleanup_temp_files() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Review & Iterate workflow stage.")
    parser.add_argument("--restart", action="store_true", help="Restart this stage from Subtask 1, clearing state for this stage.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint within this stage.")
    parser.add_argument("--iteration", type=int, default=1, help="Specify the overall model iteration number (used for tracking and output directories). Default is 1.")
    args_parsed = parser.parse_args()
    
    if args_parsed.restart and args_parsed.resume:
        print("Error: Cannot specify both --restart and --resume")
        sys.exit(1)
    
    main_args = {"iteration": args_parsed.iteration}
    if args_parsed.restart:
        main_args["restart_stage"] = REVIEW_ITERATE_STAGE
        main_args["clear_state"] = True
        asyncio.run(main(main_args))
    elif args_parsed.resume:
         # Resume will read checkpoint and determine where to start
         asyncio.run(main(main_args)) 
    else:
        # Default: Prompt user (which often implies restart if no checkpoint exists for stage)
        asyncio.run(main(main_args)) 