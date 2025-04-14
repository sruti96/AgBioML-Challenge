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
    read_arrow_file
)

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
        "perplexity_search": FunctionTool(
            query_perplexity, 
            strict=True, 
            description="Query Perplexity for an AI-powered search engine that is great for research and technical questions."
        ),
        "webpage_parser": FunctionTool(
            format_webpage, 
            strict=True, 
            description="Parse webpage content and extract readable text"
        ),
        "analyze_plot": FunctionTool(
            analyze_plot_file,
            description="""Analyze a plot file and return a description of its contents.
            You can provide a custom prompt to ask specific questions about the plot.
            If no prompt is provided, a default analysis will be performed.""",
            name="analyze_plot"
        ),
        "search_directory": FunctionTool(
            search_directory,
            description="Search for files in a specified directory, optionally filtering by pattern and searching recursively.",
            name="search_directory"
        ),
        "read_text_file": FunctionTool(
            read_text_file,
            description="Read the contents of a text file (.txt, .csv, .tsv, .json, etc.).",
            name="read_text_file"
        ),
        "read_arrow_file": FunctionTool(
            read_arrow_file,
            description="Read the contents of an Arrow file (feather format) as a pandas DataFrame.",
            name="read_arrow_file"
        )
    }
    return tools

def initialize_agents(agent_configs, tools, selected_agents=None, model_name="gpt-4o"):
    """Initialize agents based on configurations."""
    model_client = OpenAIChatCompletionClient(model=model_name)

    # Ensure all selected agents are in the agent_configs
    selected_agent_names = [config["name"] for config in agent_configs]
    if selected_agents:
        for agent_name in selected_agents:
            if agent_name not in selected_agent_names:
                raise ValueError(f"Agent {agent_name} not found in agent_configs")
    
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
        previous_summaries = previous_summaries.replace("DONE", "")
        previous_summaries = previous_summaries.replace("APPROVE", "")
        previous_summaries = previous_summaries.replace("REVISE", "")
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

# New memory management functions

def get_workflow_state():
    """Get the current workflow state (stage, subtasks completed).
    
    Returns:
        dict: The current workflow state
    """
    state_file = os.path.join(memory_dir, 'workflow_state.json')
    
    try:
        with open(state_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Default initial state
        return {
            "current_stage": 1,
            "stages_completed": [],
            "iterations": {}
        }

def update_workflow_state(stage, subtask=None, iteration=None):
    """Update the workflow state.
    
    Args:
        stage: The current stage number
        subtask: Optional subtask number
        iteration: Optional iteration number for the subtask
    """
    os.makedirs(memory_dir, exist_ok=True)
    state_file = os.path.join(memory_dir, 'workflow_state.json')
    
    state = get_workflow_state()
    
    # Update the current stage
    state["current_stage"] = stage
    
    # Update iterations if subtask is provided
    if subtask is not None:
        if "iterations" not in state:
            state["iterations"] = {}
        
        stage_key = f"stage{stage}"
        if stage_key not in state["iterations"]:
            state["iterations"][stage_key] = {}
        
        subtask_key = f"subtask{subtask}"
        if iteration is not None:
            state["iterations"][stage_key][subtask_key] = iteration
    
    with open(state_file, 'w') as f:
        json.dump(state, f)

async def save_structured_summary(stage, subtask, iteration, summary, task_description):
    """Save a summary in a structured format.
    
    Args:
        stage: The stage number
        subtask: The subtask number
        iteration: The iteration number (for subtasks that can repeat)
        summary: The summary text
        task_description: The task description
    """
    os.makedirs(memory_dir, exist_ok=True)
    summary_file = os.path.join(memory_dir, 'structured_summaries.json')
    
    try:
        with open(summary_file, 'r') as f:
            summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        summaries = {}
    
    stage_key = f"stage{stage}"
    if stage_key not in summaries:
        summaries[stage_key] = {}
    
    subtask_key = f"subtask{subtask}"
    if subtask_key not in summaries[stage_key]:
        summaries[stage_key][subtask_key] = {}
    
    # Record the timestamp for sorting later
    timestamp = datetime.datetime.now().isoformat()
    
    # Format the summary with task description
    full_summary = {
        "timestamp": timestamp,
        "iteration": iteration,
        "task_description": task_description,
        "summary": summary
    }
    
    iter_key = f"iteration{iteration}"
    summaries[stage_key][subtask_key][iter_key] = full_summary
    
    with open(summary_file, 'w') as f:
        json.dump(summaries, f)
    
    # Update the workflow state
    update_workflow_state(stage, subtask, iteration)

async def get_structured_summaries(current_stage, current_subtask=None, current_iteration=1):
    """Get structured summaries relevant to the current workflow position.
    
    Args:
        current_stage: The current stage number
        current_subtask: Optional current subtask number
        current_iteration: The current iteration number for the subtask (default: 1)
    
    Returns:
        str: Formatted summaries relevant to the current context
    """
    summary_file = os.path.join(memory_dir, 'structured_summaries.json')
    
    try:
        with open(summary_file, 'r') as f:
            all_summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No previous summaries available."
    
    result_parts = []
    
    # Add headers and separation between stages for clarity
    result_parts.append("=" * 80)
    result_parts.append("SUMMARIES FROM PREVIOUS WORKFLOW STAGES")
    result_parts.append("=" * 80)
    
    # Include summaries from all previous stages
    for stage in range(1, current_stage):
        stage_key = f"stage{stage}"
        if stage_key in all_summaries:
            stage_summaries = []
            
            result_parts.append(f"\n{'*' * 60}")
            result_parts.append(f"STAGE {stage} SUMMARIES:")
            result_parts.append(f"{'*' * 60}\n")
            
            for subtask_key, subtask_data in sorted(all_summaries[stage_key].items()):
                # For completed stages, include the final iteration of each subtask
                if not subtask_data:  # Skip empty subtasks
                    continue
                    
                iterations = [int(k.replace("iteration", "")) for k in subtask_data.keys()]
                if not iterations:  # Skip if no iterations
                    continue
                    
                final_iteration = max(iterations)  # GET THE MAXIMUM ITERATION
                final_iter_key = f"iteration{final_iteration}"
                
                if final_iter_key not in subtask_data:  # Skip if iteration doesn't exist
                    continue
                
                summary_data = subtask_data[final_iter_key]
                
                formatted_summary = f"""
SUBTASK: {subtask_key}
FINAL ITERATION: {final_iteration}

TASK DESCRIPTION:
{summary_data.get('task_description', 'No task description available')}

COMPLETED TASK RESULT:
{summary_data.get('summary', 'No summary available')}
"""
                stage_summaries.append(formatted_summary)
            
            result_parts.extend(stage_summaries)
    
    # For the current stage, include all completed subtasks
    if current_stage > 0:
        stage_key = f"stage{current_stage}"
        if stage_key in all_summaries:
            result_parts.append(f"\n{'=' * 80}")
            result_parts.append(f"CURRENT STAGE {current_stage} SUMMARIES:")
            result_parts.append(f"{'=' * 80}\n")
            
            # First, add all completed subtasks (with subtask number < current_subtask)
            for subtask_key, subtask_data in sorted(all_summaries[stage_key].items()):
                if not subtask_data:  # Skip empty subtasks
                    continue
                    
                subtask_num = int(subtask_key.replace("subtask", ""))
                
                # Include subtasks with lower numbers
                if current_subtask is not None and subtask_num < current_subtask:
                    # Get the latest iteration for this subtask
                    iterations = [int(k.replace("iteration", "")) for k in subtask_data.keys()]
                    if not iterations:  # Skip if no iterations
                        continue
                        
                    final_iteration = max(iterations)  # GET THE MAXIMUM ITERATION
                    final_iter_key = f"iteration{final_iteration}"
                    
                    if final_iter_key not in subtask_data:  # Skip if iteration doesn't exist
                        continue
                    
                    summary_data = subtask_data[final_iter_key]
                    
                    formatted_summary = f"""
SUBTASK: {subtask_key}
LATEST ITERATION: {final_iteration}

TASK DESCRIPTION:
{summary_data.get('task_description', 'No task description available')}

COMPLETED TASK RESULT:
{summary_data.get('summary', 'No summary available')}
"""
                    result_parts.append(formatted_summary)
            
            # For iterations beyond the first (current_iteration > 1), include previous iteration's later subtasks
            if current_iteration > 1:
                result_parts.append(f"\n{'=' * 80}")
                result_parts.append(f"PREVIOUS ITERATION SUMMARIES:")
                result_parts.append(f"{'=' * 80}\n")
                
                # Add all subtasks from the previous iteration
                for subtask_key, subtask_data in sorted(all_summaries[stage_key].items()):
                    if not subtask_data:  # Skip empty subtasks
                        continue
                        
                    subtask_num = int(subtask_key.replace("subtask", ""))
                    
                    # Include later subtasks from previous iteration (subtask_num > current_subtask)
                    if current_subtask is not None and subtask_num > current_subtask:
                        prev_iteration = current_iteration - 1
                        iter_key = f"iteration{prev_iteration}"
                        
                        # Only include if this iteration exists
                        if iter_key in subtask_data:
                            summary_data = subtask_data[iter_key]
                            
                            formatted_summary = f"""
SUBTASK: {subtask_key}
ITERATION: {prev_iteration}

TASK DESCRIPTION:
{summary_data.get('task_description', 'No task description available')}

COMPLETED TASK RESULT:
{summary_data.get('summary', 'No summary available')}
"""
                            result_parts.append(formatted_summary)
                
                # Also include previous iterations of the current subtask if they exist
                subtask_key = f"subtask{current_subtask}"
                if subtask_key in all_summaries[stage_key]:
                    subtask_data = all_summaries[stage_key][subtask_key]
                    for i in range(1, current_iteration):
                        iter_key = f"iteration{i}"
                        if iter_key in subtask_data:
                            summary_data = subtask_data[iter_key]
                            
                            formatted_summary = f"""
SUBTASK: {subtask_key}
ITERATION: {i}

TASK DESCRIPTION:
{summary_data.get('task_description', 'No task description available')}

COMPLETED TASK RESULT:
{summary_data.get('summary', 'No summary available')}
"""
                            result_parts.append(formatted_summary)
    
    return "\n".join(result_parts)

# Enhanced versions of existing functions that use the new structured approach

async def save_messages_structured(stage, subtask, iteration, messages, summary, task_description):
    """Save messages and summary using the structured approach.
    
    Args:
        stage: The stage number
        subtask: The subtask number
        iteration: The iteration number for this subtask
        messages: List of messages to save
        summary: The summary to save
        task_description: The task description
    """
    os.makedirs(memory_dir, exist_ok=True)
    
    # Save messages
    messages_file = os.path.join(memory_dir, 'structured_messages.json')
    try:
        with open(messages_file, 'r') as f:
            all_messages = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_messages = {}
    
    stage_key = f"stage{stage}"
    if stage_key not in all_messages:
        all_messages[stage_key] = {}
    
    subtask_key = f"subtask{subtask}"
    if subtask_key not in all_messages[stage_key]:
        all_messages[stage_key][subtask_key] = {}
    
    iter_key = f"iteration{iteration}"
    all_messages[stage_key][subtask_key][iter_key] = [msg.dump() for msg in messages]
    
    with open(messages_file, 'w') as f:
        json.dump(all_messages, f)
    
    # Save the summary
    await save_structured_summary(stage, subtask, iteration, summary, task_description)

async def format_structured_task_prompt(stage, subtask, task_text, iteration=1):
    """Format a task prompt with relevant previous summaries.
    
    Args:
        stage: The current stage number
        subtask: The current subtask number
        task_text: The current task description
        iteration: The current iteration number (default: 1)
    
    Returns:
        str: Formatted prompt with relevant previous context
    """
    previous_summaries = await get_structured_summaries(stage, subtask, iteration)
    
    if previous_summaries and previous_summaries != "No previous summaries available.":
        # Clean up any termination words
        previous_summaries = previous_summaries.replace("TERMINATE", "")
        previous_summaries = previous_summaries.replace("DONE", "")
        previous_summaries = previous_summaries.replace("APPROVE", "")
        previous_summaries = previous_summaries.replace("REVISE", "")
        
        return f"""
THESE ARE THE SUMMARIES OF PREVIOUS TASKS AND ITERATIONS. THESE PROVIDE CONTEXT FOR YOUR CURRENT TASK:

{previous_summaries}

THIS IS YOUR CURRENT TASK:

{task_text}
"""
    else:
        return f"""
THIS IS YOUR CURRENT TASK:

{task_text}
"""

# Workflow Checkpoint and Resume Functionality

def get_workflow_checkpoints():
    """Get available workflow checkpoints.
    
    Returns:
        dict: Available checkpoints with stage/subtask/iteration info
    """
    checkpoint_file = os.path.join(memory_dir, 'workflow_checkpoints.json')
    
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "stages_completed": [],
            "checkpoints": {}
        }

def save_workflow_checkpoint(stage, subtask=None, iteration=None, label=None):
    """Save a workflow checkpoint.
    
    Args:
        stage: Stage number
        subtask: Optional subtask number
        iteration: Optional iteration number
        label: Optional human-readable label for the checkpoint
    """
    os.makedirs(memory_dir, exist_ok=True)
    checkpoint_file = os.path.join(memory_dir, 'workflow_checkpoints.json')
    
    # Generate timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Generate checkpoint ID
    checkpoint_id = f"checkpoint_{stage}"
    if subtask is not None:
        checkpoint_id += f"_{subtask}"
        if iteration is not None:
            checkpoint_id += f"_{iteration}"
    checkpoint_id += f"_{timestamp.replace(':', '-').replace('.', '-')}"
    
    # Generate a human-readable description if not provided
    if label is None:
        label = f"Stage {stage}"
        if subtask is not None:
            label += f", Subtask {subtask}"
            if iteration is not None:
                label += f", Iteration {iteration}"
    
    # Get current checkpoints
    checkpoints = get_workflow_checkpoints()
    
    # Add new checkpoint
    checkpoints["checkpoints"][checkpoint_id] = {
        "timestamp": timestamp,
        "stage": stage,
        "subtask": subtask,
        "iteration": iteration,
        "label": label,
        "state": get_workflow_state()  # Save the current workflow state
    }
    
    # Update stages completed if this is a stage completion
    if subtask is None and stage not in checkpoints["stages_completed"]:
        checkpoints["stages_completed"].append(stage)
    
    # Save updated checkpoints
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoints, f, indent=2)
    
    return checkpoint_id

def mark_stage_completed(stage):
    """Mark a workflow stage as completed.
    
    Args:
        stage: Stage number to mark as completed
    """
    checkpoints = get_workflow_checkpoints()
    
    if stage not in checkpoints["stages_completed"]:
        checkpoints["stages_completed"].append(stage)
        
        with open(os.path.join(memory_dir, 'workflow_checkpoints.json'), 'w') as f:
            json.dump(checkpoints, f, indent=2)

def is_stage_completed(stage):
    """Check if a workflow stage is completed.
    
    Args:
        stage: Stage number to check
        
    Returns:
        bool: True if the stage is completed, False otherwise
    """
    checkpoints = get_workflow_checkpoints()
    return stage in checkpoints["stages_completed"]

def get_latest_checkpoint(stage=None, subtask=None):
    """Get the latest checkpoint for a stage/subtask.
    
    Args:
        stage: Optional stage number to filter by
        subtask: Optional subtask number to filter by
        
    Returns:
        dict or None: Latest checkpoint or None if not found
    """
    checkpoints = get_workflow_checkpoints()
    
    # Filter checkpoints by stage/subtask
    filtered = []
    for cp_id, cp_data in checkpoints["checkpoints"].items():
        if stage is not None and cp_data["stage"] != stage:
            continue
        if subtask is not None and cp_data["subtask"] != subtask:
            continue
        filtered.append((cp_id, cp_data))
    
    if not filtered:
        return None
    
    # Sort by timestamp and return the latest
    filtered.sort(key=lambda x: x[1]["timestamp"], reverse=True)
    return {filtered[0][0]: filtered[0][1]}

def get_maximum_iteration(stage, subtask):
    """Get the maximum iteration number for a stage/subtask.
    
    Args:
        stage: Stage number
        subtask: Subtask number
        
    Returns:
        int: Maximum iteration number, or 0 if no iterations found
    """
    summary_file = os.path.join(memory_dir, 'structured_summaries.json')
    
    try:
        with open(summary_file, 'r') as f:
            all_summaries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
    
    stage_key = f"stage{stage}"
    if stage_key not in all_summaries:
        return 0
    
    subtask_key = f"subtask{subtask}"
    if subtask_key not in all_summaries[stage_key]:
        return 0
    
    # Extract iteration numbers from keys
    iterations = [int(k.replace("iteration", "")) for k in all_summaries[stage_key][subtask_key].keys()]
    
    if not iterations:
        return 0
    
    return max(iterations)

async def clear_workflow_state(stage=None):
    """Clear the workflow state to restart from a specific stage.
    
    Args:
        stage: Optional stage number to restart from. If None, clears the entire state.
    """
    if stage is None:
        # Clear the entire workflow state
        state = {
            "current_stage": 1,
            "stages_completed": [],
            "iterations": {}
        }
    else:
        # Preserve previous stages' state but reset current and future stages
        state = get_workflow_state()
        
        # Keep only stages earlier than the specified stage
        for s in list(state.get("iterations", {}).keys()):
            s_num = int(s.replace("stage", ""))
            if s_num >= stage:
                if "iterations" in state:
                    state["iterations"].pop(s, None)
        
        state["current_stage"] = stage
    
    # Save the updated state
    os.makedirs(memory_dir, exist_ok=True)
    with open(os.path.join(memory_dir, 'workflow_state.json'), 'w') as f:
        json.dump(state, f)

async def list_available_workflow_options():
    """List available workflow options for restart/resume.
    
    Returns:
        dict: Available options with descriptions
    """
    checkpoints = get_workflow_checkpoints()
    state = get_workflow_state()
    
    options = {
        "restart": {},
        "resume": None
    }
    
    # Check for completed stages to restart from
    for stage in sorted(checkpoints["stages_completed"]):
        options["restart"][stage] = f"Restart from Stage {stage}"
    
    # Check for current state to resume
    current_stage = state.get("current_stage")
    if current_stage:
        # Find the latest checkpoint for the current stage
        latest = get_latest_checkpoint(stage=current_stage)
        if latest:
            cp_id, cp_data = list(latest.items())[0]
            subtask = cp_data.get("subtask")
            iteration = cp_data.get("iteration")
            
            resume_point = f"Stage {current_stage}"
            if subtask is not None:
                resume_point += f", Subtask {subtask}"
                if iteration is not None:
                    resume_point += f", Iteration {iteration}"
            
            options["resume"] = {
                "checkpoint_id": cp_id,
                "description": f"Resume from {resume_point}"
            }
    
    return options

async def prompt_for_workflow_action(current_stage):
    """Prompt the user for workflow action (restart, resume, start new).
    
    Args:
        current_stage: The current stage number for which we're prompting options
    
    Returns:
        dict: Action to take with details
    """
    checkpoints = get_workflow_checkpoints()
    state = get_workflow_state()
    
    options = {
        "restart": False,
        "resume": None
    }
    
    # Check if we have a resumable state for this specific stage
    has_resume_option = False
    current_stage_key = f"stage{current_stage}"
    if "iterations" in state and current_stage_key in state["iterations"]:
        # Find the latest checkpoint for this stage
        latest = get_latest_checkpoint(stage=current_stage)
        if latest:
            cp_id, cp_data = list(latest.items())[0]
            subtask = cp_data.get("subtask")
            iteration = cp_data.get("iteration")
            
            resume_point = f"Stage {current_stage}"
            if subtask is not None:
                resume_point += f", Subtask {subtask}"
                if iteration is not None:
                    resume_point += f", Iteration {iteration}"
            
            options["resume"] = {
                "checkpoint_id": cp_id,
                "description": f"Resume from {resume_point}"
            }
            has_resume_option = True
    
    # Check if prerequisite stages are completed
    prereq_ready = True
    if current_stage > 1:
        prereq_ready = is_stage_completed(current_stage - 1)
    
    print(f"\n===== Workflow Management for Stage {current_stage} =====")
    
    # Track available options for input prompt
    available_options = []
    
    # Display resume option if available
    if has_resume_option:
        print(f"R: {options['resume']['description']}")
        available_options.append("R")
    
    # Always offer restart option
    if current_stage == 1:
        print(f"S: Start Stage 1 (Understanding Phase)")
    else:
        print(f"S: Start Stage {current_stage} from the beginning" + 
              (f" (preserves Stage {current_stage-1} results)" if prereq_ready else ""))
    available_options.append("S")
    
    # Offer clean start option if we're beyond stage 1 or have existing data
    if current_stage > 1 or has_resume_option:
        print("C: Clean start (erase all previous work)")
        available_options.append("C")
    
    print("===========================")
    
    # Build the prompt string based on available options
    prompt_options = "/".join(available_options)
    choice = input(f"Enter your choice ({prompt_options}): ").strip().upper()
    
    if choice == "R" and has_resume_option:
        return {
            "action": "resume",
            "checkpoint_id": options["resume"]["checkpoint_id"]
        }
    elif choice == "S":
        if not prereq_ready and current_stage > 1:
            confirm = input(f"Stage {current_stage-1} is not completed. Proceed anyway? (y/n): ").lower()
            if confirm != 'y':
                return await prompt_for_workflow_action(current_stage)  # Ask again
        return {
            "action": "restart",
            "stage": current_stage
        }
    elif choice == "C" and (current_stage > 1 or has_resume_option):
        confirm = input("This will erase ALL previous work. Are you sure? (y/n): ").lower()
        if confirm != 'y':
            return await prompt_for_workflow_action(current_stage)  # Ask again
        return {
            "action": "new"
        }
    else:
        print(f"Invalid choice. Please enter one of: {prompt_options}")
        return await prompt_for_workflow_action(current_stage)  # Ask again

async def resume_from_checkpoint(checkpoint_id):
    """Resume workflow from a specific checkpoint.
    
    Args:
        checkpoint_id: Checkpoint ID to resume from
    
    Returns:
        dict: Restored workflow state or None if checkpoint not found
    """
    checkpoints = get_workflow_checkpoints()
    
    if checkpoint_id not in checkpoints["checkpoints"]:
        print(f"Checkpoint {checkpoint_id} not found.")
        return None
    
    # Get the checkpoint data
    checkpoint = checkpoints["checkpoints"][checkpoint_id]
    
    # Restore the workflow state
    os.makedirs(memory_dir, exist_ok=True)
    with open(os.path.join(memory_dir, 'workflow_state.json'), 'w') as f:
        json.dump(checkpoint["state"], f, indent=2)
    
    print(f"Restored workflow state from checkpoint: {checkpoint['label']}")
    print(f"Stage: {checkpoint['stage']}, Subtask: {checkpoint.get('subtask')}, Iteration: {checkpoint.get('iteration')}")
    
    return checkpoint["state"]

# Work directory management

def get_task_workdir(stage, clean=False, workdir_suffix=None):
    """Get the task-specific working directory for engineer outputs.
    
    Args:
        stage: The stage number
        clean: Whether to clean the directory if it exists
        workdir_suffix: Optional suffix to add to the workdir name
        
    Returns:
        str: Path to the task-specific working directory
    """
    # Create base name like 'task_2_workdir' or 'task_2_custom_suffix'
    if workdir_suffix:
        workdir_name = f"task_{stage}_{workdir_suffix}"
    else:
        workdir_name = f"task_{stage}_workdir"
    
    # Ensure the task workdir exists
    os.makedirs(workdir_name, exist_ok=True)
    
    # Clean the directory if requested
    if clean:
        clean_directory(workdir_name)
        
    return workdir_name

def clean_directory(directory):
    """Remove all files in a directory but keep the directory itself.
    
    Args:
        directory: Directory path to clean
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error cleaning {item_path}: {e}")
    
    print(f"Cleaned directory: {directory}")

def setup_task_environment(stage, subtask=None, is_restart=False, workdir_suffix=None):
    """Set up task environment including working directory.
    
    Args:
        stage: The stage number
        subtask: Optional subtask number 
        is_restart: Whether this is a restart (clean directory) or resume (preserve)
        workdir_suffix: Optional suffix for the workdir name
        
    Returns:
        dict: Environment details including workdir path
    """
    # Create and possibly clean the task workdir
    output_dir = get_task_workdir(stage, clean=is_restart, workdir_suffix=workdir_suffix)
    
    # Create an info file in the output directory to track details
    info = {
        "stage": stage,
        "subtask": subtask,
        "timestamp": datetime.datetime.now().isoformat(),
        "is_restart": is_restart,
        "output_directory": output_dir
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

# Task prompt loading utilities
def load_task_prompts(config_path=None):
    """Load task prompts from YAML file."""
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/tasks.yaml")
    
    with open(config_path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts.get("tasks", {})

def get_task_text(task_category, task_name, **kwargs):
    """Get a task prompt text with optional format parameters.
    
    Args:
        task_category: Category of the task (e.g., 'understanding', 'eda', 'data_split')
        task_name: Name of the specific task (e.g., 'subtask_1', 'subtask_2_revision')
        **kwargs: Format parameters to be applied to the task text
        
    Returns:
        str: The formatted task text
    """
    prompts = load_task_prompts()
    
    # Handle missing categories or task names
    if task_category not in prompts:
        print(f"Warning: Task category '{task_category}' not found in prompts.yaml")
        return ""
    
    if task_name not in prompts[task_category]:
        print(f"Warning: Task name '{task_name}' not found in category '{task_category}'")
        return ""
    
    # Get the task text and format it if kwargs are provided
    task_text = prompts[task_category][task_name].get("text", "")
    
    # If 'overall_task_text' is not explicitly provided but needed, load it from prompts
    if "overall_task_text" in task_text and "overall_task_text" not in kwargs:
        if "overall" in prompts and "text" in prompts["overall"]:
            kwargs["overall_task_text"] = prompts["overall"]["text"]
    
    if kwargs:
        try:
            task_text = task_text.format(**kwargs)
        except KeyError as e:
            print(f"Warning: Missing format parameter: {e}")
    
    return task_text

def get_system_prompt(prompt_name):
    """Get a system prompt by name from the agents.yaml file.
    
    Args:
        prompt_name: Name of the agent (e.g., 'data_science_critic', 'summarizer')
        
    Returns:
        str: The agent's system prompt text
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "config/agents.yaml")
    
    with open(config_path, "r") as f:
        agents_config = yaml.safe_load(f)
    
    if "agents" not in agents_config:
        print(f"Warning: No agents found in agents.yaml")
        return ""
    
    # Find the agent with the matching name
    for agent in agents_config["agents"]:
        if agent["name"] == prompt_name:
            return agent.get("system_prompt", "")
    
    print(f"Warning: Agent '{prompt_name}' not found in agents.yaml")
    return ""

def get_checklist(checklist_name):
    """Get a checklist by name from the tasks.yaml file.
    
    Args:
        checklist_name: Name of the checklist (e.g., 'plot_quality')
        
    Returns:
        str: The checklist text
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "config/tasks.yaml")
    
    with open(config_path, "r") as f:
        prompts = yaml.safe_load(f)
    
    if "checklists" not in prompts or checklist_name not in prompts["checklists"]:
        print(f"Warning: Checklist '{checklist_name}' not found in tasks.yaml")
        return ""
    
    return prompts["checklists"][checklist_name]

def get_agent_token(agent_configs, agent_name, token_type="termination_token"):
    """Get a token for an agent from the agent configs.
    
    Args:
        agent_configs: List of agent configurations
        agent_name: Name of the agent to find
        token_type: Type of token to retrieve (default: "termination_token")
        
    Returns:
        The token value, or None if the agent or token doesn't exist
    """
    for agent_config in agent_configs:
        if agent_config["name"] == agent_name and token_type in agent_config:
            return agent_config[token_type]
    return None
