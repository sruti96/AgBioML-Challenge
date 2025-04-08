import os
import yaml
import sys
import json

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from altum_v1.tools import query_perplexity, format_webpage, analyze_plot_file


def load_agent_configs(config_path=None):
    """Load agent configurations from YAML file."""
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/agents.yaml")
    
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs.get("agents", [])

def create_tool_instances():
    """Create instances of all available tools."""
    tools = {
        "perplexity": FunctionTool(
            query_perplexity, 
            strict=True, 
            description="Query perplexity for information on a research topic"
        ),
        "webpage_parser": FunctionTool(
            format_webpage, 
            strict=True, 
            description="Parse webpage content and extract readable text"
        ),
        # Add other tools here - these should be implemented properly
        "analyze_plot": FunctionTool(
            analyze_plot_file,
            description="""Analyze a plot file and return a description of its contents.
            You can provide a custom prompt to ask specific questions about the plot.
            If no prompt is provided, a default analysis will be performed.""",
            name="analyze_plot"
        )
    }
    return tools

def initialize_agents(agent_configs, tools, selected_agents=None, model_name="gpt-4o"):
    """Initialize agents based on configurations."""
    model_client = OpenAIChatCompletionClient(model=model_name)
    
    agents = {}
    for config in agent_configs:
        # Skip agents not in the selected list if a selection is provided
        if selected_agents and config["name"] not in selected_agents:
            continue
            
        # Get tools for this agent
        agent_tools = [
            tools[tool_name] for tool_name in config.get("tools", [])
            if tool_name in tools and tools[tool_name] is not None
        ]
        
        # Create the agent
        agents[config["name"]] = AssistantAgent(
            name=config["name"],
            system_message=config.get("system_prompt", ""),
            model_client=model_client,
            tools=agent_tools,
            model_client_stream=True,
            reflect_on_tool_use=True
        )
    
    return agents

async def load_previous_summaries() -> str:
    """Load all previous summaries and their task descriptions.
    
    Args:
        task_number: The current task number (unused, kept for backward compatibility)
        subtask_number: Optional current subtask number (unused, kept for backward compatibility)
        
    Returns:
        A formatted string containing all previous summaries and their task descriptions
    """
    memory_dir = 'memory'
    summary_file = os.path.join(memory_dir, 'all_meeting_summaries.json')
    
    try:
        with open(summary_file, 'r') as f:
            summaries = json.load(f)
            
            # Get all summaries in chronological order
            all_summaries = []
            for key in sorted(summaries.keys()):
                if summaries[key]:
                    all_summaries.extend(summaries[key])
            
            return "\n\n".join(all_summaries)
            
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "No previous summaries available."

async def save_messages(task_number: int, messages: list, summary: str, task_description: str, subtask_number: int = None):
    """Save messages and summary (with task description) to memory files.
    
    Args:
        task_number: The task number
        messages: List of messages to save
        summary: The summary to save
        task_description: The task description that generated this summary
        subtask_number: Optional subtask number. If None, saves as main task messages
    """
    memory_dir = 'memory'
    os.makedirs(memory_dir, exist_ok=True)
    
    # Save all messages
    all_messages_file = os.path.join(memory_dir, 'all_messages.json')
    try:
        with open(all_messages_file, 'r') as f:
            all_messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_messages = {}
    
    if subtask_number is None:
        key = f"task{task_number}"
    else:
        key = f"task{task_number}_subtask{subtask_number}"
    all_messages[key] = [msg.dump() for msg in messages]
    
    with open(all_messages_file, 'w') as f:
        json.dump(all_messages, f)
    
    # Save summary with task description
    summary_file = os.path.join(memory_dir, 'all_meeting_summaries.json')
    try:
        with open(summary_file, 'r') as f:
            summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        summaries = {}
    
    if subtask_number is None:
        key = f"task{task_number}_summary"
    else:
        key = f"task{task_number}_subtask{subtask_number}_summary"
    if key not in summaries:
        summaries[key] = []
    
    # Prepend task description to summary with clear boundaries
    full_summary = f"""TASK DESCRIPTION:
{task_description}

COMPLETED TASK RESULT:
{summary}"""
    summaries[key].append(full_summary)
    
    with open(summary_file, 'w') as f:
        json.dump(summaries, f)

async def format_task_prompt(task_text: str, previous_summaries: str) -> str:
    """Format a task prompt with all previous summaries and task descriptions.
    
    Args:
        task_text: The current task description
        previous_summaries: All previous summaries with their task descriptions
        
    Returns:
        Formatted prompt with all previous context before current task
    """
    if previous_summaries:
        previous_summaries = previous_summaries.replace("TERMINATE", "")
        return f"""
THESE ARE THE SUMMARIES OF ALL PREVIOUS TASKS. THESE ARE NOT THE CURRENT TASK BUT PROVIDE INFORMATION THAT MAY BE RELEVANT:

{previous_summaries}

THIS IS THE CURRENT TASK:

{task_text}
"""
    else:
        return f"""
THIS IS THE CURRENT TASK:

{task_text}
"""
