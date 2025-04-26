import os
import yaml
import datetime
import glob
import shutil

# Define memory directory as a module-level constant so it can be patched in tests
memory_dir = 'memory'

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent


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
            
        other_agents_info = "\n".join([f"{name}: {description}" for name, description in zip(agent_names, agent_descriptions) if name != name])

        # Get tools for this agent
        agent_tools = []
        for tool_name in tools.keys():
            agent_tools.append(tools[tool_name])
        
        # Use the prompt field as the system prompt
        system_prompt = config.get("prompt", "")
        # Add date context at the beginning
        system_prompt = f"CONTEXT: Today's date is {today_date}.\n\n" + system_prompt

        if system_prompt: # Add other context only if base prompt exists
            system_prompt += f"""
            The other agents on your team are: 
            {other_agents_info}

            The agents in the current conversation are:
            {', '.join(selected_agents or agent_names)}
            """
        
        # Add info about the tools available to the agent
        system_prompt += f"""
        \n\n**TOOLS**

        Whenever you encounter a question or task cannot be solved purely through reasoning, 
        or which would benefit from access to other data sources or resources,
        you should use one of the following tools to assist you:

        Here are their descriptions:
        {'\n'.join([f"{tool_name}: {tools[tool_name].description}" for tool_name in tools.keys()])}

        **YOU ARE HIGHLY ENCOURAGED TO USE TOOLS WHEN APPROPRIATE.**

        **NOTE ABOUT LAB NOTEBOOK**

        The Lab notebook is a record of all the decisions, observations, and results of the project.
        You should read from it frequently if you cannot recall observations or results from previous steps.
        You should also update with your observations, tips, heuristics, and lessons learned. This will mean
        that, in the future, you will be able to improve your performance by reusing these observations.
        
        When writing to the notebook, you should ALWAYS use the following arguments:
        - entry: <text of the entry to append>
        - entry_type: <PLAN | NOTE | OUTPUT >
        - source: <name of the agent that wrote the entry, i.e. your name>

        **NOTE ABOUT OUTPUTS**
        Whenever you generate a file, result, plot, etc. You MUST make a note of it in the notebook. 
        Specifically, you should use the OUTPUT entry type. 
        Make sure to include the file name and a description of the contents if there is a file.
        For results, make sure to describe the numerical values of results (in markdown table format).

        **NOTE ABOUT CSV FILES**
        NEVER WRITE LARGE CSV FILES. ALWAYS USE ARROW FILES INSTEAD.
        GOOD EXAMPLES
        - betas_trainsplit.arrow
        - betas_valsplit.arrow
        - betas_testsplit.arrow
        - model_evaluation.arrow
        - model_predictions.arrow
        - model_coefficients.arrow
        BAD EXAMPLES
        - betas_trainsplit.csv
        - betas_valsplit.csv
        - betas_testsplit.csv
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
def initialize_notebook(notebook_path: str):
    """Initialize the lab notebook file.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        str: Path to the initialized notebook file
    """
    
    # If the notebook doesn't exist, create it with a header
    if not os.path.exists(notebook_path):
        with open(notebook_path, 'w') as f:
            to_write = f"""
# Epigenetic Clock Development Lab Notebook
Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This notebook contains the record of experiments, decisions, and results for the 
epigenetic clock development project.

### Project Goal
Develop an accurate epigenetic clock to predict chronological age from DNA methylation data, aiming for state-of-the-art performance (Pearson correlation ≥ 0.9).

### Available Data
- `betas.arrow`: DNA methylation beta values (feature matrix)
- `metadata.arrow`: Sample metadata including chronological ages (target variable)

**Critical Note:**
- The data is was mined from several hundred DNA methylation studies. To assess generalization, the data should always be split with respect to the study of origin. For example, if conducting leave-one-out cross-validation, the "left-out" split should include a study or studies that are not present in the training set.
- There is some confounding between age, tissue, and study. This is a caveat to keep in mind when interpreting the results.

### Initial Project Planning
The project will follow a flexible workflow that can include:
1. Task understanding and data exploration
2. Exploratory data analysis (EDA)
3. Strategic data splitting for training/validation/testing
4. Model selection and development
5. Training and evaluation
6. Iterative improvement

Team A (Planning) and Team B (Implementation) will collaborate through this notebook to track progress, decisions, and results.

### Next Steps
- Team A to review the available data and formulate an initial analysis plan
- Begin with exploratory data analysis to understand feature distribution and characteristics 

# Entries

<!-- Entries will go here -->

"""
            f.write(to_write)
    
    return None

def read_notebook(notebook_path: str | None = None):
    """Read the entire lab notebook content.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses default path.
    
    Returns:
        str: The entire content of the notebook
    """
    NOTEBOOK_CHAR_LIMIT = 100_000

    if notebook_path is None:
        notebook_path = "lab_notebook.md"
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
            # Error if content is > 100k characters
            if len(content) > NOTEBOOK_CHAR_LIMIT:
                print("ERROR: Lab notebook content is too long. Please truncate it to 100,000 characters or less.")
                content = content[-NOTEBOOK_CHAR_LIMIT:]
            return content
    except FileNotFoundError:
        # Initialize the notebook if it doesn't exist
        initialize_notebook(notebook_path)
        return read_notebook(notebook_path)  # Recursively call to get the initialized content

def write_notebook(entry: str, entry_type: str = "NOTE", source: str = "SYSTEM"):
    """Append an entry to the lab notebook.
    
    Args:
        entry: The content to append to the notebook
        entry_type: Type of entry (e.g., NOTE, PLAN, OUTPUT)
        source: Name of agent that wrote the entry
    
    Returns:
        str: The entry that was appended, with metadata
    """
    
    # Format the entry with metadata
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_entry = f"\n### [{timestamp}] {source} - {entry_type}\n\n{entry}\n\n"
    
    # Append the entry to the notebook
    with open("lab_notebook.md", 'a') as f:
        f.write(formatted_entry)
    
    return formatted_entry

def format_prompt(notebook_content=None, last_team_b_output=None, task_config=None):
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

    task_config_str = yaml.dump(task_config)
    prompt_parts.append(f"TASK CONFIG: {task_config_str}")
    
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


def get_tasks_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Otherwise return the entire config
    return config
