import os
import yaml
import sys
import json
import datetime
import glob
import shutil

# Define memory directory as a module-level constant so it can be patched in tests
memory_dir = 'memory'

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altum_v1.tools import (
    query_perplexity, 
    format_webpage, 
    analyze_plot_file, 
    search_directory, 
    read_text_file,
    write_text_file,
    read_arrow_file
)

# Clean up temporary code files
def cleanup_temp_files(directory: str = "."):
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

def load_agent_configs(config_path=None):
    """Load agent configurations from YAML file."""
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/agents.yaml")
    
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def create_tool_instances():
    """Create instances of all available tools.
    
    Returns:
        dict: Dictionary of tool name -> tool instance
    """
    # Import and use get_available_tools from tools
    from tools import get_available_tools
    return get_available_tools()

def initialize_agents(agent_configs, tools, selected_agents=None, model_name="gpt-4.1"):
    """Initialize agents based on configurations."""

    # TODO: module_name should be parameterized by agents.yaml
    model_client = OpenAIChatCompletionClient(model=model_name)

    # Get current date once for this initialization
    today_date = datetime.date.today().isoformat()

    # The agent_configs is the entire dictionary from the YAML file
    # Extract the agents dictionary from it
    agents_dict = agent_configs.get("agents", {})
    
    # Get all the agent names and descriptions from the agents dictionary
    agent_names = []
    agent_descriptions = []
    
    for name, config in agents_dict.items():
        agent_names.append(name)
        # Use role as the description if available
        if "role" in config:
            agent_descriptions.append(config["role"])
        # Otherwise use name
        else:
            agent_descriptions.append(config.get("name", name))
    
    # Format these into a string
    agent_info = "\n".join([f"{name}: {description}" for name, description in zip(agent_names, agent_descriptions)])

    # Ensure all selected agents are in the agent_configs
    if selected_agents:
        for agent_name in selected_agents:
            if agent_name not in agent_names:
                raise ValueError(f"Agent {agent_name} not found in agent_configs")
    
    agents = {}
    for name, config in agents_dict.items():
        # Skip agents not in the selected list if a selection is provided
        if selected_agents and name not in selected_agents:
            continue
            
        # Get tools for this agent
        agent_tools = []
        for tool_name in config.get("tools", []):
            if tool_name in tools and tools[tool_name] is not None:
                agent_tools.append(tools[tool_name])
        
        # Use the prompt field as the system prompt
        system_prompt = config.get("prompt", "")
        # Add date context at the beginning
        system_prompt = f"CONTEXT: Today's date is {today_date}.\n\n" + system_prompt

        if system_prompt: # Add other context only if base prompt exists
            system_prompt += f"""
            The other agents on your team are: 
            {agent_info}

            The agents in the current conversation are:
            {', '.join(selected_agents or agent_names)}
            """

        agents[name] = AssistantAgent(
            name=config.get("name", name),
            system_message=system_prompt,
            model_client=model_client,
            tools=agent_tools,
            model_client_stream=True,
            reflect_on_tool_use=True
        )
    
    return agents

# Lab Notebook Functions
def initialize_notebook(notebook_path: str | None = None):
    """Initialize the lab notebook file.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        str: Path to the initialized notebook file
    """
    if notebook_path is None:
        notebook_path = os.path.join(memory_dir, 'lab_notebook.md')
    
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    
    # If the notebook doesn't exist, create it with a header
    if not os.path.exists(notebook_path):
        with open(notebook_path, 'w') as f:
            f.write(f"# Epigenetic Clock Development Lab Notebook\n")
            f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"This notebook contains the record of experiments, decisions, and results for the epigenetic clock development project.\n\n")
            f.write(f"## Entries\n\n")
    
    return notebook_path

def read_notebook(notebook_path: str | None = None):
    """Read the entire lab notebook content.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        str: The entire content of the notebook
    """
    if notebook_path is None:
        notebook_path = os.path.join(memory_dir, 'lab_notebook.md')
    
    try:
        with open(notebook_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Initialize the notebook if it doesn't exist
        initialize_notebook(notebook_path)
        return read_notebook(notebook_path)  # Recursively call to get the initialized content

def write_notebook(entry: str, entry_type: str = "NOTE", team: str = "SYSTEM", notebook_path: str | None = None):
    """Append an entry to the lab notebook.
    
    Args:
        entry: The content to append to the notebook
        entry_type: Type of entry (e.g., NOTE, PLAN, RESULT, METRIC)
        team: Source of the entry (e.g., SYSTEM, TEAM_A, TEAM_B)
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        str: The entry that was appended, with metadata
    """
    if notebook_path is None:
        notebook_path = os.path.join(memory_dir, 'lab_notebook.md')
    
    # Ensure the notebook exists
    if not os.path.exists(notebook_path):
        initialize_notebook(notebook_path)
    
    # Format the entry with metadata
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_entry = f"\n### [{timestamp}] {team} - {entry_type}\n\n{entry}\n\n"
    
    # Append the entry to the notebook
    with open(notebook_path, 'a') as f:
        f.write(formatted_entry)
    
    return formatted_entry

def format_prompt(overall_task, notebook_content=None, last_team_b_output=None, task_config=None):
    """Format a prompt with notebook content and last Team B output.
    
    Args:
        overall_task: The overall task description
        notebook_content: Optional content of lab notebook
        last_team_b_output: Optional last output from Team B
        task_config: Optional task configuration from tasks.yaml
        
    Returns:
        str: Formatted prompt with all context
    """
    prompt_parts = []
    
    # Add the overall task
    prompt_parts.append("# OVERALL TASK")
    prompt_parts.append(overall_task)
    prompt_parts.append("")
    
    # Add project context from task_config if available
    if task_config:
        if "project_goal" in task_config:
            prompt_parts.append("# PROJECT GOAL")
            prompt_parts.append(task_config["project_goal"])
            prompt_parts.append("")
        
        if "project_context" in task_config:
            prompt_parts.append("# PROJECT CONTEXT")
            prompt_parts.append(task_config["project_context"])
            prompt_parts.append("")
            
        if "available_data" in task_config:
            prompt_parts.append("# AVAILABLE DATA")
            for data_item in task_config["available_data"]:
                prompt_parts.append(f"- {data_item.get('name', '')}: {data_item.get('description', '')}")
            prompt_parts.append("")
            
        if "autonomous_workflow" in task_config:
            workflow = task_config["autonomous_workflow"]
            if "approach" in workflow:
                prompt_parts.append("# WORKFLOW APPROACH")
                prompt_parts.append(workflow["approach"])
                prompt_parts.append("")
            
            if "methodology" in workflow:
                prompt_parts.append("# METHODOLOGY GUIDELINES")
                prompt_parts.append(workflow["methodology"])
                prompt_parts.append("")
                
        if "lab_notebook_guidelines" in task_config:
            prompt_parts.append("# LAB NOTEBOOK GUIDELINES")
            prompt_parts.append(task_config["lab_notebook_guidelines"])
            prompt_parts.append("")
    
    # Add notebook content if provided
    if notebook_content:
        prompt_parts.append("# LAB NOTEBOOK CONTENT")
        prompt_parts.append(notebook_content)
        prompt_parts.append("")
    
    # Add last Team B output if provided
    if last_team_b_output:
        prompt_parts.append("# LATEST IMPLEMENTATION REPORT FROM TEAM B")
        prompt_parts.append(last_team_b_output)
        prompt_parts.append("")
    
    # Add instruction for Team A
    prompt_parts.append("# YOUR CURRENT TASK")
    prompt_parts.append("Review the lab notebook and the latest implementation (if any). Based on the overall goal and current progress:")
    prompt_parts.append("1. Discuss the current state of the project.")
    prompt_parts.append("2. Identify the next logical step to advance the project.")
    prompt_parts.append("3. Create a detailed specification for Team B to implement this next step.")
    prompt_parts.append("4. The Principal Scientist should summarize the discussion and provide the final specification.")
    
    return "\n".join(prompt_parts)

def get_task_text(task_category, task_name, config_path=None, full_task=False, **kwargs):
    """Get a task prompt text or full task structure with optional format parameters.
    
    Args:
        task_category: Category of the task (e.g., 'overall')
        task_name: Name of the specific task (e.g., 'text')
        config_path: Optional path to tasks.yaml file
        full_task: If True, returns the entire task dictionary instead of just the text field
        **kwargs: Format parameters to be applied to the task text (only for text field)
        
    Returns:
        str or dict: The formatted task text or full task dictionary
    """
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/tasks.yaml")
    
    with open(config_path, "r") as f:
        prompts = yaml.safe_load(f)
    
    # Handle missing categories or task names
    if task_category not in prompts.get("tasks", {}):
        print(f"Warning: Task category '{task_category}' not found in tasks.yaml")
        return {} if full_task else ""
    
    if task_name not in prompts["tasks"][task_category]:
        print(f"Warning: Task name '{task_name}' not found in category '{task_category}'")
        return {} if full_task else ""
    
    # Return the entire task dictionary if full_task is True
    if full_task:
        return prompts["tasks"][task_category][task_name]
    
    # Otherwise, get just the text field and format it if kwargs are provided
    task_text = prompts["tasks"][task_category][task_name].get("text", "")
    
    if kwargs:
        try:
            task_text = task_text.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing format parameter: {e}")
    
    return task_text

def get_agent_token(agent_configs, agent_name, token_type="termination_token"):
    """Get a token for an agent from the agent configs.
    
    Args:
        agent_configs: List of agent configurations
        agent_name: Name of the agent to find
        token_type: Type of token to retrieve (default: "termination_token")
        
    Returns:
        The token value, or None if the agent or token doesn't exist
    """
    agents_dict = agent_configs.get("agents", {})
    agent_config = agents_dict.get(agent_name, {})
    return agent_config.get(token_type)

def setup_task_environment(base_dir=None, is_restart=False, iteration=None):
    """Set up task environment including working directory.
    
    Args:
        base_dir: Base directory for outputs
        is_restart: Whether to clean the directory
        iteration: Optional iteration number for subdirectory 
        
    Returns:
        dict: Environment details including workdir path
    """
    # Default base directory
    if base_dir is None:
        base_dir = f"task_outputs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory (with optional iteration subdirectory)
    if iteration is not None:
        output_dir = os.path.join(base_dir, f"iteration_{iteration}")
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the directory if requested
    if is_restart and os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error cleaning {item_path}: {e}")
    
    # Create an info file to track details
    info = {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "timestamp": datetime.datetime.now().isoformat(),
        "iteration": iteration,
        "is_restart": is_restart
    }
    
    with open(os.path.join(output_dir, "task_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    # Handle data files - check for common data files in current directory
    common_data_files = ['betas.arrow', 'metadata.arrow']
    current_dir = os.getcwd()
    
    # Track the found files and their locations
    data_files_info = {}
    
    # Check current directory for data files
    for filename in common_data_files:
        if os.path.exists(os.path.join(current_dir, filename)):
            data_files_info[filename] = {
                "location": "current_dir",
                "path": os.path.join(current_dir, filename),
                "relative_path": filename
            }
        else:
            data_files_info[filename] = {
                "location": "not_found",
                "status": "File not found in accessible directories"
            }
    
    # Add data files info to the environment details
    info["data_files"] = data_files_info
    info["docker_working_directory"] = current_dir
    
    # Update the info file with data files information
    with open(os.path.join(output_dir, "task_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    return {
        "workdir": current_dir,            # The Docker container working directory
        "output_dir": output_dir,          # Where to save outputs
        "info": info,
        "data_files": data_files_info
    }

def get_tasks_config(config_path=None, section=None):
    """Load and return the tasks configuration, either the entire file or a specific section.
    
    Args:
        config_path: Optional path to tasks.yaml file
        section: Optional specific section to return (e.g., 'project_goal', 'reference', etc.)
        
    Returns:
        dict: The entire tasks configuration or the requested section
    """
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/tasks.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Return specific section if requested
    if section:
        if section in config:
            return config[section]
        else:
            print(f"Warning: Section '{section}' not found in tasks.yaml")
            return {}
    
    # Otherwise return the entire config
    return config
