# Altum v2: Autonomous Agent-based Workflow

## Overview

Altum v2 is a reimagined, streamlined version of the original Altum workflow for building an epigenetic clock using DNA methylation data. Unlike the original version, which used a rigid, multi-stage workflow across multiple scripts, Altum v2 uses a single script with autonomous agents to determine the next logical steps in the scientific process.

## Key Features

- **Single Script Execution**: The entire workflow runs in one script (`01_run_pipeline_v2.py`).
- **Agent Autonomy**: Agents decide what steps to take next rather than following a predetermined workflow.
- **Team-Based Approach**:
  - **Team A (Planning)**: Principal Scientist, ML Expert, and Bioinformatics Expert discuss and create plans.
  - **Team B (Engineering Society)**: Engineer, Code Executor, and Data Science Critic implement and evaluate the plans.
- **Lab Notebook**: Long-term memory is implemented via a lab notebook that both teams can read and write to.
- **Iterative Development**: The system follows an iterative cycle where Team A plans, Team B implements, and results feed back into planning.

## Directory Structure

- `01_run_pipeline.py`: Main workflow script
- `utils.py`: Utility functions for file operations, notebook handling, etc.
- `tools.py`: Tool implementations (Perplexity search, file operations, etc.)
- `agents.py`: Custom agent implementations (TeamAPlanning, EngineerSocietyV2)
- `config/`: Configuration files
  - `agents.yaml`: Agent configurations
  - `tasks.yaml`: Task descriptions
- `memory/`: Storage for laboratory notebook and other memory
- `task_outputs/`: Output directory for results

## Running the Pipeline

```bash
python experiments/altum_v2/01_run_pipeline.py [options]
```

### Options

- `--max-iterations`: Maximum number of Team A-B exchanges (default: 10)
- `--output-dir`: Base directory for outputs (default: task_outputs)
- `--notebook-path`: Path to the lab notebook (default: memory_v2/lab_notebook.txt)

## Implementation Details

### Agent Architecture

1. **TeamAPlanning**: A meta-agent that coordinates the Principal Scientist, ML Expert, and Bioinformatics Expert in a round-robin chat. The Principal Scientist makes final decisions.

2. **EngineerSocietyV2**: A meta-agent that coordinates the Engineer and Code Executor, with the Data Science Critic reviewing their work. The cycle continues until the Critic approves.

3. **Main Group Chat**: A round-robin chat between Team A and Team B, where Team A provides plans and Team B implements them.

### Memory Management

- **Short-term memory**: Preserved by the agents' message history.
- **Long-term memory**: Implemented as a lab notebook in a text file, which both teams can read from and write to.

### Tool System

- Tools are shared across agents based on their configuration.
- Both teams have access to file operations, search capabilities, and notebook operations.

## Design Principles

1. **Autonomy with Guidance**: Agents have freedom to determine next steps but follow best practices embedded in their prompts.
2. **Iterative Development**: Teams cycle through plan-implement-evaluate loops.
3. **Scientific Documentation**: Important decisions, results, and metrics are documented in the lab notebook for reference.
4. **Streamlined Execution**: Single script execution makes it easy to run and monitor the entire workflow.

## Differences from Altum v1

1. Single script vs. multiple stage-specific scripts
2. Autonomous determination of next steps vs. fixed stage progression
3. Lab notebook as long-term memory vs. stage-based summaries
4. Message passing between teams vs. summarization
5. Direct implementation of feedback without the need for manual stage transitions 