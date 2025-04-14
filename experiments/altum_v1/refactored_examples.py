"""Examples of refactored code using the new prompts.yaml and agent.py files."""

import os
import asyncio
from time import sleep

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

from altum_v1.utils import (
    initialize_agents,
    load_agent_configs,
    create_tool_instances,
    format_structured_task_prompt,
    save_messages_structured,
    get_task_text,
    get_system_prompt,
    get_checklist,
    save_workflow_checkpoint,
    mark_stage_completed,
    update_workflow_state
)
from altum_v1.agents import EngineerSociety

# Constants for workflow stages 
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3

# Example of refactored run_understanding_task function from 01_understand_problem.py
async def run_understanding_task_refactored():
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
    
    # Get the task text from prompts.yaml
    task_text = get_task_text("understanding", "task_1")
    
    # Format the task with structured format
    formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
    
    # Run the agent group on the task
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
    
    # Save checkpoint after completion
    save_workflow_checkpoint(stage, subtask, iteration, "Understanding Complete")
    
    # Update workflow state to mark understanding stage as completed
    mark_stage_completed(UNDERSTANDING_STAGE)
    
    # Create checkpoint for the start of the next stage
    save_workflow_checkpoint(EDA_STAGE, label="Ready for EDA")
    
    return result

# Example of refactored run_subtask_2 function from 03_split_data.py
async def run_subtask_2_refactored(iteration=1, task_env=None, retry_count=0):
    """Run the second subtask: Engineer implementing the data splitting specification."""
    # Update workflow state
    stage = DATA_SPLIT_STAGE
    subtask = 2
    
    try:
        # Clean up any temp files from previous runs if retrying
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of MAX_RETRIES...")
            # cleanup_temp_files()  # Implement or import this function
            # Short pause to ensure cleanup is complete
            sleep(1)
        
        # Initialize agents and tools
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        
        # Make sure search_directory tool is available
        from altum_v1.tools import search_directory
        if 'search_directory' not in available_tools:
            available_tools['search_directory'] = search_directory
        
        # Initialize model client for all agents
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        
        # 1. Create the engineer agent
        which_agents = ['engineer']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)
        engineer_agent = list(agents.values())[0]
        
        # 2. Create the code executor agent
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=task_env["workdir"],
            timeout=600
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
        
        # 3. Create the critic agent using system prompt from prompts.yaml
        critic_agent = AssistantAgent(
            name="data_science_critic",
            system_message=get_system_prompt("critic"),
            model_client=model_client,
            tools=[tool for tool in available_tools.values()],
            model_client_stream=True,
            reflect_on_tool_use=True
        )
        
        # 4. Create the summarizer agent using system prompt from prompts.yaml
        summarizer_agent = AssistantAgent(
            name="technical_summarizer",
            system_message=get_system_prompt("summarizer"),
            model_client=model_client,
            model_client_stream=True
        )
        
        # Define termination conditions
        engineer_termination = TextMentionTermination("ENGINEER_DONE")
        critic_termination = TextMentionTermination("TERMINATE_CRITIC")
        
        # Create the engineer team (engineer + code executor)
        engineer_team = RoundRobinGroupChat(
            participants=[engineer_agent, code_executor_agent],
            termination_condition=engineer_termination,
            max_turns=100  # Allow enough turns for multiple iterations
        )
        
        # Create the critic team
        critic_team = RoundRobinGroupChat(
            participants=[critic_agent],
            termination_condition=critic_termination,
            max_turns=25  # Allow multiple turns for ReAct cycles
        )
        
        # Get the appropriate task text based on iteration
        if iteration == 1:
            task_text = get_task_text("data_split", "subtask_2")
        else:
            task_text = get_task_text("data_split", "subtask_2_revision", iteration=iteration)
        
        # Append directory information to the task text
        task_text += f"\n\nIMPORTANT FILE ORGANIZATION INSTRUCTIONS:"
        task_text += f"\n- Your code runs in the main project directory where data files are located"
        task_text += f"\n- SAVE ALL OUTPUT FILES to the '{task_env['output_dir']}' directory"
        task_text += f"\n- This includes plots, intermediate data files, and any other outputs"
        task_text += f"\n- Example: plt.savefig('{task_env['output_dir']}/my_plot.png')"
        task_text += f"\n- The split data files should be saved as train.arrow, val.arrow, test.arrow and train_meta.arrow, val_meta.arrow, test_meta.arrow"
        
        # Add data format guardrail
        task_text += f"\n\n⚠️ CRITICAL DATA FORMAT REQUIREMENT ⚠️"
        task_text += f"\n- ALWAYS save dataframes ONLY in arrow/feather format using .to_feather() or similar methods"
        task_text += f"\n- NEVER save data to CSV, TSV, or any other text-based format as these are highly inefficient"
        task_text += f"\n- Example: df.to_feather('{task_env['output_dir']}/data.arrow')"
        task_text += f"\n- For any intermediate files, also use arrow/feather format"
        task_text += f"\n- If you need to export small amounts of summary statistics, use JSON or pickle, but not CSV"
        
        # Add information about plot quality requirement
        task_text += f"\n\nANALYSIS QUALITY VERIFICATION REQUIREMENT:"
        task_text += f"\n- A plot quality checklist has been saved to '{task_env['output_dir']}/plot_quality_checklist.txt'"
        task_text += f"\n- For your analysis, you MUST:"
        task_text += f"\n  1. Provide detailed TABULAR STATISTICS for all key metrics within the text of your analysis report"
        task_text += f"\n  2. Include cross-tabulations showing counts AND percentages for categorical variables"
        task_text += f"\n  3. Use statistical tests to verify the representativeness of splits"
        task_text += f"\n  4. Present tables BEFORE visualizations as your primary evidence"
        task_text += f"\n  5. Use visualizations to support and enhance the tabular statistics"
        
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
        
        # Format the task with previous context
        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        # Create the task message
        task_message = TextMessage(
            content=formatted_task,
            source="User"
        )
        
        # Create the EngineerSociety that manages the interaction between teams
        engineer_society = EngineerSociety(
            name="data_splitting_society",
            engineer_team=engineer_team,
            critic_team=critic_team,
            critic_approve_token="APPROVE_ENGINEER",
            engineer_terminate_token="ENGINEER_DONE",
            critic_terminate_token="TERMINATE_CRITIC",
            critic_revise_token="REVISE_ENGINEER",
            summarizer_agent=summarizer_agent,
            original_task=task_text
        )
        
        # Run the engineer society with the formatted task
        print(f"Starting EngineerSociety execution for data splitting (iteration {iteration})...")
        result = await engineer_society.on_messages([task_message], CancellationToken())
        
        # Extract messages for saving
        engineer_messages = [task_message]  # Start with the task message
        if result.inner_messages:
            engineer_messages.extend(result.inner_messages)
        if result.chat_message:
            engineer_messages.append(result.chat_message)
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, engineer_messages, 
                                     result.chat_message.content if result.chat_message else "No result", 
                                     task_text)
        
        # Save the summarized report for subtask 3 to use
        if result.chat_message and hasattr(result.chat_message, 'content'):
            summary_file_path = os.path.join(task_env['output_dir'], f"implementation_summary_iteration_{iteration}.txt")
            with open(summary_file_path, "w") as f:
                f.write(result.chat_message.content)
            print(f"Saved implementation summary to {summary_file_path}")
        
        return result
        
    except Exception as e:
        print(f"Error in subtask 2 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # If we haven't exhausted retries, try again
        if retry_count < 3:  # MAX_RETRIES - 1
            print(f"Retrying subtask 2 (iteration {iteration})...")
            return await run_subtask_2_refactored(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries exceeded for subtask 2. Giving up.")
            return None 