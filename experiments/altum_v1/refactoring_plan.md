# Refactoring Plan for Flock-AG Codebase

## 1. Main Areas for Refactoring

### 1.1 Agent Definitions
- Move all agent-specific system prompts from Python code to the `agents.yaml` file
- This includes:
  - `CRITIC_SYSTEM_PROMPT` 
  - `SUMMARIZER_SYSTEM_PROMPT`
  - All other agent prompts defined in Python files

### 1.2 Task Prompts
- Create a new `prompts.yaml` file to store all task prompts
- Move the following from Python files to YAML:
  - `OVERALL_TASK_TEXT`
  - `SUBTASK_1_TEXT`, `SUBTASK_2_TEXT`, `SUBTASK_3_TEXT`
  - `SUBTASK_2_REVISION_TEXT`
  - `PLOT_QUALITY_CHECKLIST`
  - Other instructional text that's currently duplicated

### 1.3 Agent Initialization
- Standardize agent initialization across all scripts
- Use the `initialize_agents` function consistently with proper agent configurations

### 1.4 Custom Agent Classes
- Move custom agent implementations like `EngineerSociety` to a separate module
- Create a reusable implementation that can be imported in multiple scripts

## 2. Specific Implementation Tasks

### 2.1 Create Task Prompts YAML
```yaml
# Create experiments/altum_v1/config/prompts.yaml
tasks:
  overall:
    text: >
      Your goal is to build an epigenetic clock...
  understanding:
    task_1:
      text: >
        # CURRENT WORKFLOW STEP: Step 1 - Understanding the task...
  eda:
    subtask_1:
      text: >
        # CURRENT WORKFLOW STEP: Task 2, Subtask 1 - Exploratory Data Analysis...
    subtask_2:
      text: >
        # CURRENT WORKFLOW STEP: Task 2, Subtask 2 - Exploratory Data Analysis...
    subtask_2_revision:
      text: >
        # CURRENT WORKFLOW STEP: Task 2, Subtask 2 - Exploratory Data Analysis (REVISION)...
    subtask_3:
      text: >
        # CURRENT WORKFLOW STEP: Task 2, Subtask 3 - Exploratory Data Analysis (Review and Iteration)...
  data_split:
    subtask_1:
      text: >
        # CURRENT WORKFLOW STEP: Task 3, Subtask 1 - Data Splitting Strategy...
    subtask_2:
      text: >
        # CURRENT WORKFLOW STEP: Task 3, Subtask 2 - Data Splitting Implementation...
    subtask_2_revision:
      text: >
        # CURRENT WORKFLOW STEP: Task 3, Subtask 2 - Data Splitting Implementation (REVISION)...
    subtask_3:
      text: >
        # CURRENT WORKFLOW STEP: Task 3, Subtask 3 - Data Splitting Review...
  checklists:
    plot_quality: >
      # CRITICAL PLOT QUALITY CHECKLIST...
```

### 2.2 Update `agents.yaml`
- Add the critic and summarizer agent system prompts to the YAML file
- Move all other agent-specific prompts from Python files to YAML

### 2.3 Create Agent Module
- Create `experiments/altum_v1/agents.py` file to house custom agent implementations

### 2.4 Create Task Prompt Loading Utilities
```python
# Add to utils.py
def load_task_prompts(config_path=None):
    """Load task prompts from YAML file."""
    if config_path is None:
        # Use path relative to current file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  "config/prompts.yaml")
    
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
    if task_category in prompts and task_name in prompts[task_category]:
        task_text = prompts[task_category][task_name].get("text", "")
        if kwargs:
            task_text = task_text.format(**kwargs)
        return task_text
    return ""
```

### 2.5 Refactor Python Script Files
- Update `01_understand_problem.py`, `02_eda.py`, and `03_split_data.py` to:
  - Remove hardcoded task texts
  - Use `get_task_text` to load task prompts
  - Use standardized agent initialization
  - Import custom agent implementations from the new module

## 3. Implementation Sequence

1. Create `config/prompts.yaml` with all task prompts
2. Update `agents.yaml` with all agent system prompts
3. Create the `agents.py` module with custom agent implementations
4. Add task prompt loading utilities to `utils.py`
5. Refactor each script file one by one to use the new structure
6. Test each refactored script to ensure functionality is preserved
7. Remove duplicate code and clean up the codebase 