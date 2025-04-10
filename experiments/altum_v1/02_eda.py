# This team will have two components:
# A. A team of agents that discuss the EDA and generate a specification for the engineer
# B. An engineer and code execution agent that writes the code to carry out the specification and returns the results
# The workflow is:
# Subtask 1: The team A agents discuss the EDA and generate a specification for the engineer
# Subtask 2: The team B agents write and execute the code to carry out the specification. They return the results in a report.
# Subtask 3: The team A agents review the report and generate a new specification if necessary.
# Subtask 2 and 3 continue until the principal scientist is satisfied with the results and terminates the workflow.

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
    setup_task_environment
)

load_dotenv()

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
EXPERIMENTAL_DESIGN_STAGE = 3
MODEL_TRAINING_STAGE = 4

# Maximum number of retries for each subtask
MAX_RETRIES = 3

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

SUBTASK_1_TEXT="""
# CURRENT WORKFLOW STEP: Task 2, Subtask 1 - Exploratory Data Analysis (Team A Discussion)

# TEAM COMPOSITION:
# - Principal Scientist: Team leader who makes key research decisions and determines when discussions should conclude
# - Bioinformatics Expert: Specialist in omics data processing with deep knowledge of DNA methylation data handling
# - Machine Learning Expert: Provides insights on ML architectures, feature engineering, and model evaluation approaches

Your team has been provided with DNA methylation data and metadata files. As Team A, your task is to:

1. Review the previous understanding phase report and identify key areas that need exploration in the data
2. Develop a comprehensive EDA specification that includes:
   - Data quality assessment plan
   - Distribution analysis requirements
   - Feature correlation analysis needs
   - Batch effect investigation strategy
   - Visualization requirements
   - Statistical tests to perform
3. Consider the following aspects in your specification:
   - Biological relevance of the analyses
   - Technical feasibility of implementation
   - Computational efficiency
   - Potential pitfalls and edge cases
   - Validation approaches

Your output should be a detailed specification document that Team B can implement. It MUST include 
STEP-BY-STEP instructions for the engineer to follow!
The Principal Scientist will summarize the discussion and provide the final specification when the team reaches consensus.
OF NOTE: Team B only has access to python, not R. Do NOT suggest using any R packages in your specification.

Remember to:
- Build on each other's expertise
- Consider both biological and technical aspects
- Be specific about analysis requirements
- Justify your recommendations
- Consider practical implementation constraints
"""

SUBTASK_2_TEXT="""
# CURRENT WORKFLOW STEP: Task 2, Subtask 2 - Exploratory Data Analysis (Implementation)

# TEAM COMPOSITION:
# - ML/Data Engineer: Implements the technical solutions in code based on the team's plans
# - Code Executor: Executes the code and returns the results.

As the ML/Data Engineer, you have received the EDA specification from Team A. Your task is to:

1. Review the specification document carefully
2. Implement the requested analyses using the provided tools:
   - Code runner with pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
   - Local file system access
3. Create a comprehensive report that includes:
   - Code implementation details
   - Analysis results
   - Visualizations
   - Key findings
   - Potential issues or limitations
   - Recommendations for next steps

Your report should be:
- Well-documented with clear explanations
- Include all relevant code and outputs
- Highlight important patterns and insights
- Note any technical challenges encountered
- Suggest potential improvements or additional analyses

Remember to:
- Follow best practices in scientific computing
- Include appropriate error handling
- Optimize for performance with large datasets
- Make code reusable and maintainable
- Document all implementation choices
"""

SUBTASK_2_REVISION_TEXT="""
# CURRENT WORKFLOW STEP: Task 2, Subtask 2 - Exploratory Data Analysis (REVISION)

# TEAM COMPOSITION:
# - ML/Data Engineer: Implements the technical solutions in code based on the team's plans
# - Code Executor: Executes the code and returns the results.

# IMPORTANT: This is ITERATION {iteration} of this subtask. You are expected to address the feedback provided by Team A in their review.

As the ML/Data Engineer, you have received FEEDBACK on your previous implementation from Team A. Your task is to:

1. Review Team A's feedback carefully and acknowledge each point
2. Implement the requested revisions and additional analyses
3. Address all the issues and suggestions raised by the review team
4. Enhance your previous implementation based on their guidance
5. Create a revised comprehensive report that includes:
   - Response to each feedback point
   - Updated code implementation
   - New or enhanced visualizations
   - Additional analyses requested
   - Revised findings and insights

Your revised report should:
- Begin with a clear acknowledgment of the feedback points you're addressing
- Explicitly reference how each implementation change responds to the feedback
- Include improved visualizations and analyses as requested
- Highlight new insights discovered through the revisions
- Note any ongoing challenges or limitations

Remember to:
- Maintain high-quality code with appropriate comments
- Continue following computational efficiency guidelines
- Ensure all visualizations are properly saved and documented
- Address ALL the feedback points from Team A
- Provide a complete, standalone report that incorporates the revisions
"""

SUBTASK_3_TEXT="""
# CURRENT WORKFLOW STEP: Task 2, Subtask 3 - Exploratory Data Analysis (Review and Iteration)

# TEAM COMPOSITION:
# - Principal Scientist: Team leader who makes key research decisions and determines when discussions should conclude
# - Bioinformatics Expert: Specialist in omics data processing with deep knowledge of DNA methylation data handling
# - Machine Learning Expert: Provides insights on ML architectures, feature engineering, and model evaluation approaches

Team A, you have received the implementation report from the ML/Data Engineer. Your task is to:

1. Review the implementation and results thoroughly
2. Assess whether the specification was fully implemented
3. Evaluate the quality and completeness of the analyses
4. Identify any gaps or areas needing further exploration
5. Determine if additional analyses are required

IMPORTANT: The engineer has generated several plots that are available for your analysis. You can:
- Use the analyze_plot tool to examine any plot mentioned in the report
- Ask specific questions about patterns, distributions, or anomalies
- Request additional visualizations if needed
- Discuss the implications of the visual findings

IMPORTANT: As a best practice, you should ALWAYS request at least one round of revision to ensure thorough analysis. Only approve the results if you are absolutely certain that no further improvements are needed.

Based on your review, you should:
- Acknowledge successful aspects of the implementation
- Identify any missing or incomplete analyses
- Suggest improvements or additional analyses if needed
- Consider the implications of the findings for the next steps
- Analyze and discuss the visualizations provided
- Request additional plots if key aspects are not visualized

If additional analyses are needed:
- Provide a new, focused specification
- Be specific about what needs to be done differently
- Justify the need for additional analyses
- Consider computational efficiency
- Specify any additional visualizations required

The Principal Scientist will summarize the review and either:
- Conclude the EDA phase if satisfied by saying "APPROVE"
- Request additional analyses if needed by saying "REVISE"

Remember to:
- Be constructive in your feedback
- Consider both biological and technical aspects
- Focus on actionable improvements
- Maintain scientific rigor
- Make use of the available visualizations in your analysis
"""

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

    # Create agent group for the discussion task
    text_termination = TextMentionTermination("TERMINATE")
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination
    )
    
    # Format the task with previous context
    formatted_task = await format_structured_task_prompt(stage, subtask, SUBTASK_1_TEXT, iteration)
    
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], SUBTASK_1_TEXT)
    
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
        which_agents = ['engineer']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=task_env["workdir"],  # Use main working directory for Docker access
            timeout=600
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)

        # Create agent group for the implementation task
        text_termination = TextMentionTermination("DONE")
        task_group = RoundRobinGroupChat(
            participants=list(agents.values()) + [code_executor_agent],
            termination_condition=text_termination
        )
        
        # Choose the appropriate task text based on iteration
        if iteration == 1:
            task_text = SUBTASK_2_TEXT
        else:
            # Format the revision text with the current iteration number
            task_text = SUBTASK_2_REVISION_TEXT.format(iteration=iteration)
        
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
        
        result = await Console(task_group.run_stream(task=formatted_task))
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
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

# Similar updates for run_subtask_3
async def run_subtask_3(iteration=1, task_env=None, retry_count=0):
    """Run the third subtask: Team reviewing the implementation.
    
    Args:
        iteration: The iteration number for this subtask
        task_env: Optional task environment including workdir
        retry_count: Current retry attempt (0 = first attempt)
    """
    # Update workflow state
    stage = EDA_STAGE
    subtask = 3
    update_workflow_state(stage, subtask, iteration)
    
    # Get or create task environment if not provided
    if task_env is None:
        # For subtask 3, we want to use the same workdir as subtask 2
        task_env = setup_task_environment(stage, subtask, is_restart=False)
    
    try:
        # Clean up any temp files from previous runs
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {MAX_RETRIES}...")
            cleanup_temp_files()
            # Short pause to ensure cleanup is complete
            time.sleep(1)
            
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        which_agents = ['principal_scientist', 'bioinformatics_expert', 'ml_expert']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

        # Create agent group for the review task
        text_termination = TextMentionTermination("TERMINATE")
        task_group = RoundRobinGroupChat(
            participants=list(agents.values()),
            termination_condition=text_termination
        )
        
        # Add information about the output directory to the task text
        task_text = f"{SUBTASK_3_TEXT}\n\nNOTE: The engineer's output files, including all plots and generated files, can be found in the '{task_env['output_dir']}' directory."
        
        # Format the task with previous context, including the current iteration
        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        result = await Console(task_group.run_stream(task=formatted_task))
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
        return result
        
    except Exception as e:
        print(f"Error in subtask 3 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        
        # If we haven't exhausted retries, try again
        if retry_count < MAX_RETRIES - 1:
            print(f"Retrying subtask 3 (iteration {iteration})...")
            return await run_subtask_3(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries ({MAX_RETRIES}) exceeded for subtask 3. Giving up.")
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
                            
                            # Continue with subtask 3
                            result3 = await run_subtask_3(current_iteration, task_env)
                            if not result3:
                                print(f"Subtask 3 (iteration {current_iteration}) failed to complete")
                                return
                            
                            # Check if approved
                            last_message = result3.messages[-1].dump()["content"]
                            
                            # Continue iterations or finish based on result
                            if "APPROVE" in last_message:
                                print(f"EDA completed and approved after {current_iteration} iterations")
                                mark_stage_completed(EDA_STAGE)
                                return
                            elif "REVISE" in last_message:
                                # Run the iteration loop below
                                pass
                            else:
                                raise ValueError("Last message from principal scientist is not 'APPROVE' or 'REVISE'")
                        elif highest_subtask == 3:  # Completed subtask 3, check result
                            # Need to check if approved or needs revision
                            # This would need to read the stored summary
                            return  # Need more specific handling here
                        
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
    
    # Run subtask 3: Team reviewing the implementation
    result3 = await run_subtask_3(iteration, task_env)
    if not result3:
        print("Subtask 3 failed to complete after multiple retries")
        return
    
    # Save checkpoint after subtask 3
    save_workflow_checkpoint(EDA_STAGE, 3, iteration, f"EDA Review (Iteration {iteration})")
    
    # Check if additional iterations are needed
    last_message = result3.messages[-1].dump()["content"]
    
    # Continue iterations until approved
    while "APPROVE" not in last_message:
        if "REVISE" not in last_message:
            raise ValueError("Last message from principal scientist is not 'APPROVE' or 'REVISE'")
        
        # Clean up temporary files between iterations
        cleanup_temp_files()
        
        # Increment iteration counter
        iteration += 1
        
        # Run subtask 2 again with new iteration number
        result2 = await run_subtask_2(iteration, task_env)
        if not result2:
            print(f"Subtask 2 (iteration {iteration}) failed to complete after multiple retries")
            return
        
        # Save checkpoint after subtask 2 (new iteration)
        save_workflow_checkpoint(EDA_STAGE, 2, iteration, f"EDA Implementation (Iteration {iteration})")
        
        # Run subtask 3 again with new iteration number
        result3 = await run_subtask_3(iteration, task_env)
        if not result3:
            print(f"Subtask 3 (iteration {iteration}) failed to complete after multiple retries")
            return
        
        # Save checkpoint after subtask 3 (new iteration)
        save_workflow_checkpoint(EDA_STAGE, 3, iteration, f"EDA Review (Iteration {iteration})")
        
        # Check if approved this time
        last_message = result3.messages[-1].dump()["content"]
    
    # If we've reached here, the EDA was approved
    print(f"EDA completed and approved after {iteration} iterations")
    
    # Final cleanup of temporary files
    cleanup_temp_files()
    
    # Update workflow state to mark EDA stage as completed
    mark_stage_completed(EDA_STAGE)
    save_workflow_checkpoint(EXPERIMENTAL_DESIGN_STAGE, label="Ready for Experimental Design")

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





