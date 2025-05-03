import os
import sys
import asyncio
import argparse
import datetime
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path


from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from docker.types import DeviceRequest
from autogen_agentchat.agents import CodeExecutorAgent

# Local imports 
# Set up script directory for imports
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)



# Constants
MAX_ITERATIONS = 25  # Maximum number of Team A-B exchanges
NOTEBOOK_PATH = "lab_notebook.md"  # Path to the lab notebook (now relative to working dir)
DEFAULT_WORKING_DIR = "workdir"  # Default working directory

async def main(args):
    """
    Main function to run the Altum pipeline.
    
    Args:
        args: Command-line arguments
    """
    print("Starting AltumAge V2 pipeline")

    original_dir = os.getcwd()
    
    # Resolve source and working directories
    working_dir = Path(args.working_dir).resolve()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = working_dir / f"run_{run_id}"
    
    print(f"Working directory: {working_dir}")

    # Create working directory if it doesn't exist
    working_dir.mkdir(exist_ok=True, parents=True)

    # Source dir is the location of this script
    source_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Add multiple paths to Python path for imports
    for path in [str(working_dir), str(source_dir), str(original_dir)]:
        if path not in sys.path:
            sys.path.append(path)
            sys.path.append(os.path.join(path, "experiments"))

    # Local imports
    from utils import (
        load_agent_configs,
        create_tool_instances,
        initialize_agents,
        initialize_notebook,
        read_notebook,
        write_notebook,
        get_tasks_config,
        format_prompt,
        get_agent_token,
        cleanup_temp_files
    )
    from agents import TeamAPlanning, EngineerSociety

    # Copy data files from source to working directory if they don't exist there already
    data_files = ['betas.arrow', 'metadata.arrow']
    for file in data_files:
        src_file = source_dir / file
        dest_file = working_dir / file
        
        if src_file.exists() and not dest_file.exists():
            print(f"Copying {file} from source directory to working directory...")
            shutil.copy2(src_file, dest_file)
        elif not dest_file.exists():
            print(f"Warning: {file} not found in source directory or working directory")
        
    
    # Define paths relative to working directory
    notebook_path = working_dir / NOTEBOOK_PATH
    
    # Initialize lab notebook in working directory if it doesn't exist
    initialize_notebook(notebook_path)
    
    # Switch working directory for all operations
    original_dir = os.getcwd()
    os.chdir(working_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    
    try:        
        # Config path is the location of this script / config directory
        config_path = Path(os.path.dirname(os.path.abspath(__file__))) / "config"
        # Confirm files exist
        if not os.path.exists(config_path / "agents.yaml"):
            raise FileNotFoundError("Could not find agents.yaml in config directory")
        if not os.path.exists(config_path / "tasks.yaml"):
            raise FileNotFoundError("Could not find tasks.yaml in config directory")
            
        agent_configs = load_agent_configs(config_path=config_path / "agents.yaml")
        available_tools = create_tool_instances()
        
        # Get the overall task description and full task configuration
        tasks_config_path = config_path / "tasks.yaml"
        task_config = get_tasks_config(config_path=tasks_config_path)
        
        # Initialize the agents needed for Team A
        team_a_agents = ['principal_scientist', 'bioinformatics_expert', 'ml_expert']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=team_a_agents, tools=available_tools)
        
        # Get the principal scientist's termination token
        principal_scientist_termination_token = get_agent_token(agent_configs, "principal_scientist")
        
        # Initialize TeamAPlanning
        team_a = TeamAPlanning(
            name="team_a_planning",
            principal_scientist=agents['principal_scientist'],
            ml_expert=agents['ml_expert'],
            bioinformatics_expert=agents['bioinformatics_expert'],
            principal_scientist_termination_token=principal_scientist_termination_token,
            max_turns=15
        )
        
        # Initialize engineer team for TeamB
        engineer_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['implementation_engineer'], tools=available_tools)['implementation_engineer']
        engineer_termination_token = get_agent_token(agent_configs, "implementation_engineer")
        
        # Set up code executor for the engineer
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=os.getcwd(),  # Use current (working) directory
            timeout=3600,
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
        
        # Create the engineer team as a round-robin group chat
        from autogen_agentchat.conditions import TextMentionTermination
        engineer_team = RoundRobinGroupChat(
            participants=[engineer_agent, code_executor_agent],
            termination_condition=TextMentionTermination(engineer_termination_token),
            max_turns=75
        )
        
        # Initialize critic team
        critic_agent = initialize_agents(agent_configs=agent_configs, selected_agents=['data_science_critic'], tools=available_tools)['data_science_critic']
        critic_termination_token = get_agent_token(agent_configs, "data_science_critic")
        critic_team = RoundRobinGroupChat(
            participants=[critic_agent],
            termination_condition=TextMentionTermination(critic_termination_token)
        )
        
        # Get tokens for EngineerSociety
        critic_approve_token = get_agent_token(agent_configs, "data_science_critic", "approval_token")
        critic_revise_token = get_agent_token(agent_configs, "data_science_critic", "revision_token")
        
        # Initialize EngineerSociety
        team_b = EngineerSociety(
            name="team_b_engineering",
            engineer_team=engineer_team,
            critic_team=critic_team,
            critic_approve_token=critic_approve_token,
            engineer_terminate_token=engineer_termination_token,
            critic_terminate_token=critic_termination_token,
            critic_revise_token=critic_revise_token,
            max_messages_to_return=25
        )
        
        # Run the main loop
        iteration = 0
        last_team_b_output = None
        project_complete = False
        
        while iteration < args.max_iterations and not project_complete:
            iteration += 1
            print(f"Starting iteration {iteration}/{args.max_iterations}")
            
            # Step 1: Team A Planning Phase
            print(f"Iteration {iteration}: Team A Planning Phase")
            
            # Read current notebook content
            notebook_content = read_notebook(notebook_path)
            
            # Format the prompt for Team A with current context
            team_a_prompt = format_prompt(
                notebook_content=notebook_content,
                last_team_b_output=last_team_b_output,
                task_config=task_config
            )
            
            # Run Team A to get next steps plan
            team_a_message = TextMessage(content=team_a_prompt, source="User")
            team_a_response = await team_a.on_messages([team_a_message], CancellationToken())
            
            # Check if the project is complete by detecting the "ENTIRE_TASK_DONE" token
            if "ENTIRE_TASK_DONE" in team_a_response.chat_message.content:
                print("DETECTED PROJECT COMPLETION TOKEN. ALL REQUIREMENTS HAVE BEEN MET.")
                project_complete = True
                write_notebook(
                    entry="PROJECT COMPLETED: All requirements have been satisfied. The Principal Scientist has verified completion.",
                    entry_type="COMPLETION",
                    source="principal_scientist"
                )
            
            # Log and record Team A's plan
            print(f"Team A planning complete for iteration {iteration}")
            
            # Make sure to remove the termination token AFTER checking for ENTIRE_TASK_DONE
            plan_content = team_a_response.chat_message.content
            
            write_notebook(
                entry=plan_content,
                entry_type="PLAN",
                source="principal_scientist"
            )
            
            # If the project is complete, skip Team B implementation
            if project_complete:
                print("Project is complete. Skipping Team B implementation.")
                break
            
            # Step 2: Team B Implementation Phase
            print(f"Iteration {iteration}: Team B Implementation Phase")
            
            # Forward Team A's plan to Team B
            team_b_message = team_a_response.chat_message
            team_b_response = await team_b.on_messages([team_b_message], CancellationToken())
            
            # Store Team B's output for next iteration
            last_team_b_output = team_b_response.chat_message.content
            critic_review = team_b_response.inner_messages[-1]
            
            # Clean up temporary files
            cleanup_temp_files()
            
            # Log completion of iteration
            print(f"Completed iteration {iteration}/{args.max_iterations}")
            
            # Write iteration summary to notebook
            write_notebook(
                entry=f"Completed iteration {iteration}. Team B critic summary:\n\n{critic_review.content}",
                entry_type="OUTPUT",
                source="TEAM_B - Implementation Summary",
            )
        
        # Close the code executor
        await code_executor.stop()
        
        print("Altum pipeline completed")
    
    finally:
        # Restore original working directory
        os.chdir(original_dir)
        print(f"Restored original working directory: {os.getcwd()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Altum pipeline")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, 
                        help=f"Maximum number of Team A-B exchanges (default: {MAX_ITERATIONS})")
    parser.add_argument("--working-dir", type=str, default=DEFAULT_WORKING_DIR,
                        help=f"Working directory for the pipeline (default: {DEFAULT_WORKING_DIR})")
    
    args = parser.parse_args()
    
    # Update constants based on arguments
    MAX_ITERATIONS = args.max_iterations
    
    asyncio.run(main(args)) 