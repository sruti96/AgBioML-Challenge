# This team will have two components:
# A. A team of agents that discuss the EDA and generate a specification for the engineer
# B. An engineer and code execution agent that writes the code to carry out the specification and returns the results
# The workflow is:
# Subtask 1: The team A agents discuss the EDA and generate a specification for the engineer
# Subtask 2: The team B agents write and execute the code to carry out the specification. They return the results in a report.
# Subtask 3: The team A agents review the report and generate a new specification if necessary.
# Subtask 2 and 3 continue until the principal scientist is satisfied with the results and terminates the workflow.

import asyncio

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
    format_task_prompt,
    load_previous_summaries,
    save_messages
)

load_dotenv()

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
# CURRENT WORKFLOW STEP: Step 2 - Exploratory Data Analysis (Team A Discussion)

Your team has been provided with DNA methylation data and metadata files. As Team A (Senior Advisor, Principal Scientist, Bioinformatics Expert, and Machine Learning Expert), your task is to:

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
# CURRENT WORKFLOW STEP: Step 2 - Exploratory Data Analysis (Implementation)

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

SUBTASK_3_TEXT="""
# CURRENT WORKFLOW STEP: Step 2 - Exploratory Data Analysis (Review and Iteration)

Team A, you have received the implementation report from the ML/Data Engineer. Your task is to:

1. Review the implementation and results thoroughly
2. Assess whether the specification was fully implemented
3. Evaluate the quality and completeness of the analyses
4. Identify any gaps or areas needing further exploration
5. Determine if additional analyses are required

Based on your review, you should:
- Acknowledge successful aspects of the implementation
- Identify any missing or incomplete analyses
- Suggest improvements or additional analyses if needed
- Consider the implications of the findings for the next steps

If additional analyses are needed:
- Provide a new, focused specification
- Be specific about what needs to be done differently
- Justify the need for additional analyses
- Consider computational efficiency

The Principal Scientist will summarize the review and either:
- Conclude the EDA phase if satisfied by saying "APPROVE"
- Request additional analyses if needed by saying "REVISE"

Remember to:
- Be constructive in your feedback
- Consider both biological and technical aspects
- Focus on actionable improvements
- Maintain scientific rigor
"""

async def run_subtask_1():
    """Run the first subtask."""
    # Load previous summaries
    previous_summaries = await load_previous_summaries()
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['principal_scientist', 'bioinformatics_expert', 'machine_learning_expert']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    # Create agent group for the understanding task
    text_termination = TextMentionTermination("TERMINATE")
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination
    )
    
    # Format the task with previous context
    formatted_task = await format_task_prompt(SUBTASK_1_TEXT, previous_summaries)
    
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description
    await save_messages(2, result.messages, result.messages[-1].dump()["content"], SUBTASK_1_TEXT, subtask_number=1)
    
    return result

async def run_subtask_2():
    """Run the second subtask."""
    # Load previous summaries
    previous_summaries = await load_previous_summaries()
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['engineer']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    code_executor = DockerCommandLineCodeExecutor(
        image='agenv:latest',
        work_dir='./tmp',
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
    
    # Format the task with previous context
    formatted_task = await format_task_prompt(SUBTASK_2_TEXT, previous_summaries)
    
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description
    await save_messages(2, result.messages, result.messages[-1].dump()["content"], SUBTASK_2_TEXT, subtask_number=2)
    
    return result

async def run_subtask_3():
    """Run the third subtask."""
    # Load previous summaries
    previous_summaries = await load_previous_summaries()
    
    agent_configs = load_agent_configs()
    available_tools = create_tool_instances()
    which_agents = ['principal_scientist', 'bioinformatics_expert', 'machine_learning_expert']
    agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

    # Create agent group for the review task
    text_termination = TextMentionTermination("TERMINATE")
    task_group = RoundRobinGroupChat(
        participants=list(agents.values()),
        termination_condition=text_termination
    )
    
    # Format the task with previous context
    formatted_task = await format_task_prompt(SUBTASK_3_TEXT, previous_summaries)
    
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description
    await save_messages(2, result.messages, result.messages[-1].dump()["content"], SUBTASK_3_TEXT, subtask_number=3)
    
    return result

async def main():
    """Run the complete EDA workflow."""
    # # Run subtask 1
    # result1 = await run_subtask_1()
    # if not result1:
    #     print("Subtask 1 failed to complete")
    #     return
    
    # Run subtask 2
    result2 = await run_subtask_2()
    if not result2:
        print("Subtask 2 failed to complete")
        return
    
    # Run subtask 3
    result3 = await run_subtask_3()
    if not result3:
        print("Subtask 3 failed to complete")
        return
    
    # Check if additional iterations are needed
    last_message = result3.messages[-1].dump()["content"]
    if "APPROVE" not in last_message:
        if "REVISE" not in last_message:
            raise ValueError("Last message is not 'APPROVE' or 'REVISE'")
        # If not terminated, continue the iteration
        await main()

if __name__ == "__main__":
    asyncio.run(main())





