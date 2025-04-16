# This team will have two components:
# A. A team of agents that discuss the data splitting approach and generate a specification for the engineer
# B. An engineer and code execution agent that writes the code to carry out the specification and returns the results
# The workflow is:
# Subtask 1: The team A agents discuss the data splitting approach and generate a specification for the engineer
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
import json
import datetime
from time import sleep

from dotenv import load_dotenv

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent, SocietyOfMindAgent, AssistantAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_core import CancellationToken
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from typing import Sequence

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
    setup_task_environment,
    resume_from_checkpoint
)
from altum_v1.tools import search_directory

load_dotenv()

# Write a function to estimate the number of tokens in a list of messages
def estimate_tokens(messages):
    """Estimate the number of tokens in a list of messages."""
    # Get the total number of words in the messages
    total_words = sum(len(message.content.split()) for message in messages)
    # Get the total number of tokens in the messages
    total_tokens = total_words * 3
    return total_tokens



# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3  # Updated name
MODEL_TRAINING_STAGE = 4

# Maximum number of retries for each subtask
MAX_RETRIES = 3

# ReAct-specific instruction for the critic's self-dialogue
CRITIC_REACT_INSTRUCTION = """
For this review task, follow the ReAct process (Reasoning, Action, Observation) strictly.

# YOUR ROLE: 
You are ONLY a reviewer. Your job is to find issues and tell the engineer to fix them with "REVISE_ENGINEER".
DO NOT try to write code or implement solutions yourself.

STEPS:
1. FIRST, use search_directory to find all visualization files:
   - Run search_directory("task_3_workdir", "*.png") to find all PNG files
   - This will give you the exact filenames to analyze
   - If no files are found, search the current directory with search_directory(".", "*.png")

2. After finding the files, analyze each one:
   - For each plot file found, use analyze_plot with the exact path
   - Example: analyze_plot("task_3_workdir/age_distribution.png")
   - Document any issues with each visualization

3. For statistical claims:
   - Verify if numbers match what's shown in plots
   - Check if methodology follows requirements

4. ONLY after examining all available plots and claims:
   - Provide a comprehensive assessment
   - List specific improvements needed
   - End with appropriate termination commands

5. If plots cannot be found or analyzed:
   - IMMEDIATELY provide feedback with "REVISE_ENGINEER" followed by "TERMINATE_CRITIC"
   - Do NOT try to create the plots yourself
   - Do NOT write example code - that's the engineer's job

# TERMINATION COMMANDS:

You MUST end your review with BOTH an action command AND a termination command:

1. For reviews with issues (including the first review):
   REVISE_ENGINEER
   TERMINATE_CRITIC

2. ONLY when all requirements are fully met:
   APPROVE_ENGINEER
   TERMINATE_CRITIC

# CRITICAL REQUIREMENTS:

1. FIRST REVIEW: You MUST find issues and end with "REVISE_ENGINEER" followed by "TERMINATE_CRITIC"

2. MISSING FILES: If ANY required files are missing, respond with "REVISE_ENGINEER" followed by "TERMINATE_CRITIC"

3. NEVER IMPLEMENT: Do NOT write code to fix problems - that's not your job

4. PROPER TERMINATION: You MUST include BOTH the action ("REVISE_ENGINEER" or "APPROVE_ENGINEER") AND "TERMINATE_CRITIC"

5. DO NOT USE CODE BLOCKS: Never use triple backticks (```) in your responses, as they can be misinterpreted as code

CRITICAL: Always use search_directory to find files before trying to analyze them. Never approve without analyzing all plot files.
"""

# System prompt for the summarizer agent
SUMMARIZER_SYSTEM_PROMPT = """
You are a technical report writer who specializes in creating clear, concise summaries of data science implementations.

Your task is to create a comprehensive final report based on:
1. The original task description that was given to the engineer
2. The series of messages from the engineer showing their implementation progress

# YOUR ROLE:
As a summarizer, you must:
- Extract the key components of the data splitting implementation
- Document the methodology used by the engineer
- Highlight the main findings and visualizations
- Organize the information in a structured, easy-to-understand format
- Create a standalone report that Team A can review without needing to read the entire implementation discussion

# REPORT STRUCTURE:
Your report should include:

## 1. Executive Summary
- Brief overview of the task and approach (1-2 paragraphs)
- Key findings and outcomes

## 2. Implementation Methodology
- Dataset characteristics
- Data splitting approach used
- Stratification techniques employed
- Key parameters and settings

## 3. Results and Validation
- Summary of dataset splits (sizes, distributions)
- Visualization descriptions (what was plotted and what it showed)
- Statistical validation performed
- Evidence of representativeness across splits

## 4. Technical Details
- List of all generated output files
- List of all generated visualizations
- Any important implementation notes

## 5. Conclusion
- Summary of the data splitting results
- Any limitations or considerations for future work

# IMPORTANT GUIDELINES:
- Focus on FACTS, not opinions
- Be precise about numerical results and methodology details
- Use clear, technical language
- Maintain scientific objectivity
- Include specific details (file paths, dataset sizes, etc.)
- Do not invent or assume information that wasn't provided in the engineer's messages
- When in doubt about a detail, use language like "approximately" or "according to the implementation"

Your report will be provided to Team A as the official documentation of the engineer's work.
"""

class EngineerSociety(BaseChatAgent):
    """A custom agent that manages the interaction between an engineer team and a critic team.
    
    This replaces the previous SocietyOfMindAgent implementation with a more direct approach
    that cycles between the engineer team and the critic team until the critic approves.
    """
    def __init__(self, name: str, engineer_team: RoundRobinGroupChat, critic_team: RoundRobinGroupChat, 
                 critic_approve_token: str, engineer_terminate_token: str, critic_terminate_token: str, 
                 critic_revise_token: str, summarizer_agent=None, original_task=None) -> None:
        super().__init__(name, description="An agent that performs data splitting implementation with critical feedback.")
        self._engineer_team = engineer_team
        self._critic_team = critic_team
        self._engineer_terminate_token = engineer_terminate_token
        self._critic_terminate_token = critic_terminate_token
        self._critic_approve_token = critic_approve_token
        self._critic_revise_token = critic_revise_token
        self._summarizer_agent = summarizer_agent
        self._original_task = original_task
        self.messages_to_summarize = []  # Track all engineer messages

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages through the engineer team and critic team with a single round of review.
        
        The flow is:
        1. Engineer team (engineer + executor) writes and runs code
        2. Critic team reviews the results
        3. Result is summarized and returned regardless of critic approval
        """
        NUM_LAST_MESSAGES = 50
        original_messages = messages
        print(f"TOKEN ESTIMATE: engineer society: {estimate_tokens(messages)}")
        print(f"NUM MESSAGES: {len(messages)}")
        sleep(5)
        # Run the engineer team with the given messages
        result_engineer = await Console(self._engineer_team.run_stream(task=messages, cancellation_token=cancellation_token))
        

        engineer_messages = result_engineer.messages
        engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
        engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
        print(f"TOKEN ESTIMATE: engineer team: {estimate_tokens(engineer_messages)}")
        print(f"NUM MESSAGES: {len(engineer_messages)}")
        sleep(5)
        # in the last message, remove the engineer_terminate_token
        engineer_messages[-1].content = engineer_messages[-1].content.replace(self._engineer_terminate_token, "")
        if len(engineer_messages) > NUM_LAST_MESSAGES:
            last_messages_engineer = engineer_messages[-NUM_LAST_MESSAGES:]
        else:
            last_messages_engineer = engineer_messages
        
        # Store the last message from the engineer
        self.messages_to_summarize.extend(last_messages_engineer)
        
        last_message_critic = None
        revision_counter = 0
        while True:
            # Run the critic team with the updated messages
            if last_message_critic is not None:
                messages_for_critic = original_messages + [last_message_critic] + last_messages_engineer
            else:
                messages_for_critic = original_messages + last_messages_engineer
            print(f"TOKEN ESTIMATE: critic team before run {revision_counter}: {estimate_tokens(messages_for_critic)}")
            print(f"NUM MESSAGES: {len(messages_for_critic)}")
            sleep(5)
            result_critic = await Console(self._critic_team.run_stream(task=messages_for_critic, cancellation_token=cancellation_token))
            critic_messages = result_critic.messages
            
            critic_messages = [message for message in critic_messages if isinstance(message, TextMessage)]
            print(f"TOKEN ESTIMATE: critic team after run {revision_counter}: {estimate_tokens(critic_messages)}")
            print(f"NUM MESSAGES: {len(critic_messages)}")
            sleep(5)
            # in the last message, remove the critic_terminate_token
            critic_messages[-1].content = critic_messages[-1].content.replace(self._critic_terminate_token, "")
            critic_messages[-1].content = critic_messages[-1].content.replace(self._critic_revise_token, "")
            last_message_critic = critic_messages[-1]
            self.messages_to_summarize.append(last_message_critic)
            
            # Check if critic approves the work
            if self._critic_approve_token in last_message_critic.content:
                break
            else:
                revision_counter += 1
                if revision_counter > 3:
                    break

            # Run the engineer team with the updated messages
            print(f"TOKEN ESTIMATE: engineer team before run {revision_counter}: {estimate_tokens(last_messages_engineer + [last_message_critic])}")
            print(f"NUM MESSAGES: {len(last_messages_engineer + [last_message_critic])}")
            sleep(5)
            result_engineer = await Console(self._engineer_team.run_stream(task=original_messages + last_messages_engineer + [last_message_critic], cancellation_token=cancellation_token))
            engineer_messages = result_engineer.messages
            
            engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
            engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
            # engineer_messages = [message for message in engineer_messages if message.source == "engineer"]
            print(f"TOKEN ESTIMATE: engineer team after run {revision_counter}: {estimate_tokens(engineer_messages)}")
            print(f"NUM MESSAGES: {len(engineer_messages)}")
            sleep(5)
            # in the last message, remove the engineer_terminate_token
            engineer_messages[-1].content = engineer_messages[-1].content.replace(self._engineer_terminate_token, "")
            if len(engineer_messages) > NUM_LAST_MESSAGES:
                last_messages_engineer = engineer_messages[-NUM_LAST_MESSAGES:]
            else:
                last_messages_engineer = engineer_messages
            self.messages_to_summarize.extend(last_messages_engineer)
            
        # Generate summary report if a summarizer agent is provided
        if self._summarizer_agent and self._original_task:
            summary_content = f"""
# Original Task:
{self._original_task}

# Engineer Implementation and Critic Feedback:
{self._format_message_history()}
"""
            breakpoint()
            summary_message = TextMessage(content=summary_content, source="User")
            print(f"TOKEN ESTIMATE: summarizer agent: {estimate_tokens([summary_message])}")
            print(f"NUM MESSAGES: {len([summary_message])}")
            sleep(5)
            summary_result = await self._summarizer_agent.on_messages([summary_message], cancellation_token)
            final_result = summary_result.chat_message
            
            # Add a note about where the original implementation details can be found
            if isinstance(final_result, TextMessage):
                final_result.content += "\n\n(Note: This is a summary of the engineer's implementation and the critic's feedback. The full implementation details and code can be found in the previous messages.)"
        else:
            # If no summarizer agent, just return the last engineer message
            final_result = last_messages_engineer
            
        return Response(chat_message=final_result, inner_messages=engineer_messages + critic_messages)
    
    def _format_message_history(self):
        """Format the history of engineer and critic messages for summarization."""
        formatted_history = ""
        for i, message in enumerate(self.messages_to_summarize):
            iteration_num = i + 1
            formatted_history += f"\n\n==== ITERATION {iteration_num} ====\n"
            formatted_history += f"\nMessage_source: {message.source}\n"
            formatted_history += message.content
        return formatted_history

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        # Reset the inner teams
        await self._engineer_team.reset()
        await self._critic_team.reset()

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)







# Critic system prompt for plot quality assessment
CRITIC_SYSTEM_PROMPT = """
You are a data science expert who reviews code implementation, methodology, statistical rigor, and presentation of results.

Follow the ReAct framework: Reasoning, Action, Observation for your analysis.

# YOUR ROLE:
You are a reviewer/critic of the overall data science implementation. Your role is to:
1. EVALUATE the methodology and statistical soundness of the implementation
2. ASSESS the quality of evidence provided (both tabular and visual)
3. PROVIDE constructive feedback to help improve the analysis
4. COMMUNICATE these issues to the engineer with "REVISE_ENGINEER"

# CRITICAL: Your primary focus is on METHODOLOGY and EVIDENCE, not just visuals.

Your analysis should assess:
1. The VALIDITY of the data splitting methodology
2. The QUALITY of tabular statistics and summary information
3. The STATISTICAL VERIFICATION of all claims made
4. The SUPPORTING visualizations (as a secondary concern)

# ANALYSIS PROCESS:
For each aspect of the implementation, follow this structure:

## REASONING:
- Identify the key methodological aspects that need evaluation
- Consider what constitutes sufficient statistical evidence
- Formulate specific questions about the approach and results
- Determine if the tabular summaries provide adequate information

## ACTION:
- FIRST use search_directory to locate all output files
- Carefully examine the tabular statistics provided
- Use analyze_plot tool to examine visualizations as supporting evidence
- Consider how well the implementation addresses the requirements

## OBSERVATION:
- Document your findings from each aspect of the analysis
- Note specific issues with methodology, statistics, or evidence
- Assess whether improvements have been made since previous feedback

# EVALUATION CRITERIA:

1. METHODOLOGY:
   - Is the data splitting approach statistically sound?
   - Are the chosen methods appropriate for the data characteristics?
   - Is the implementation rigorous and complete?
   - Are there any logical flaws or missing steps?

2. STATISTICAL EVIDENCE:
   - Are there comprehensive tabular summaries of the data splits?
   - Do the tables include counts, percentages, and statistical tests?
   - Is the evidence sufficient to validate the approach?
   - Are cross-tabulations provided for key categorical variables?

3. VISUALIZATION SUPPORT (secondary focus):
   - Do the visualizations effectively support the main findings?
   - Are they clear, accurate, and appropriately labeled?
   - Do they add value beyond what's shown in the tables?

# TERMINATING YOUR DIALOGUE:
You must END your review with one of these two termination statements:

1. When SPECIFIC ISSUES are found:
   REVISE_ENGINEER
   TERMINATE_CRITIC

2. When requirements are met OR no specific issues can be identified:
   APPROVE_ENGINEER
   TERMINATE_CRITIC

# CRITICAL APPROVAL REQUIREMENTS:

1. You MUST say "APPROVE_ENGINEER" (followed by "TERMINATE_CRITIC") if:
   - The implementation has no significant flaws
   - The method follows all critical requirements
   - Any minor issues aren't substantial enough to require revisions
   - Your assessment is generally positive

2. You should ONLY say "REVISE_ENGINEER" when you can identify SPECIFIC, ACTIONABLE issues:
   - You MUST list at least 2-3 concrete problems that need addressing
   - Each issue must be specific and actionable (not vague or general)
   - You cannot request revisions without identifying specific problems

3. NEVER request revisions if you only have general or vague feedback
   - If you find yourself making general positive statements followed by "REVISE_ENGINEER", you should use "APPROVE_ENGINEER" instead
   - Do not request revisions just to see if the engineer can make minor improvements

4. For the first iteration ONLY:
   - You are required to find at least three specific issues with the implementation
   - These must be concrete, actionable items the engineer can address
   - If you cannot find three specific issues, focus on visualization improvements

Until SIGNIFICANT issues are resolved, you must respond with detailed, constructive feedback in this format:

# ASSESSMENT:
[Provide your overall assessment of the implementation]

# SPECIFIC ISSUES REQUIRING REVISION:
1. [Specific issue #1 with clear description of what needs to be fixed]
2. [Specific issue #2 with clear description of what needs to be fixed]
3. [Specific issue #3 with clear description of what needs to be fixed]

# REQUIRED IMPROVEMENTS:
[Specify exactly what needs to be improved with concrete examples]

REVISE_ENGINEER
TERMINATE_CRITIC

Remember, your goal is to ensure methodological soundness and sufficient statistical evidence, primarily through tabular summaries backed by appropriate visualizations. Do not request revisions unless you can specify exactly what needs to be fixed.
"""

# Plot quality checklist
PLOT_QUALITY_CHECKLIST = """
# CRITICAL PLOT QUALITY CHECKLIST
Before finalizing any data visualization, verify that each plot meets these requirements:

1. READABILITY:
   - All axis labels are clearly visible and properly sized
   - Legend is readable and correctly represents the data
   - Title clearly explains what the plot shows
   - Text is not overlapping or cut off
   - Font sizes are consistent and appropriately sized

2. ACCURACY:
   - Data is correctly represented (no missing categories)
   - Color scheme properly differentiates between categories
   - Proportions and distributions match the underlying data
   - No data points are accidentally excluded
   - Scales are appropriate for the data range

3. INTERPRETABILITY:
   - The key insight is immediately apparent 
   - Comparisons between groups are clear
   - For crowded categorical plots:
     * Consider using horizontal orientation for long category names
     * Use proper spacing or faceting to avoid overcrowding
     * Ensure all categories are readable (rotate labels if needed)
   
4. VISUAL VERIFICATION:
   - After creating each plot, you MUST:
     * Explicitly print summary statistics that confirm what the plot shows
     * Compare these statistics with what's visually represented
     * Check for any discrepancies between the numbers and the visualization
   
5. QUALITY CONTROL STEPS:
   - For each visualization, run this verification code:
     * Print the exact count and percentage of items in each category
     * Verify these numbers match what's shown in the plot
     * If any category appears missing or incorrectly sized, recreate the plot

"""

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
# CURRENT WORKFLOW STEP: Task 3, Subtask 1 - Data Splitting Strategy (Team A Discussion)

# TEAM COMPOSITION:
# - Principal Scientist: Team leader who makes key research decisions and determines when discussions should conclude
# - Bioinformatics Expert: Specialist in omics data processing with deep knowledge of DNA methylation data handling
# - Machine Learning Expert: Provides insights on ML architectures, feature engineering, and model evaluation approaches

Your team has completed the Exploratory Data Analysis (EDA) phase for the DNA methylation data. Now it's time to create 
a data splitting strategy. As Team A, your task is to:

1. Review the findings from the EDA phase and identify key considerations for splitting the data
2. Develop a comprehensive data splitting specification that includes:
   - Splitting ratios for train/validation/test sets
   - Stratification strategy (if any)
   - Handling of potential batch effects or confounding variables
   - Balance considerations for important variables (age distribution, tissue types, sex, etc.)
   - Evaluation criteria to ensure the splits are representative
   - Preprocessing or transformations to apply before or after splitting
3. Consider the following aspects in your specification:
   - Biological relevance of the splitting approach
   - Technical feasibility of implementation
   - Impact on model training and evaluation
   - Potential biases and how to address them
   - Validation approaches to ensure split quality

# CRITICAL BIOLOGICAL DATA SPLITTING GUIDELINES
When splitting biological data, you MUST consider the following hierarchical sources of technical variance (from largest to smallest effect):

1. Study/Dataset Level:
   - Studies represent a data collection effort, typically performed by a single research group with their own equipment, personnel, and protocols
   - Different studies have different protocols and equipment, which can introduce technical bias
   - If multiple studies/datasets exist, the test set SHOULD contain data from studies NOT represented in train/val
   - Often a major source of technical variance that can lead to overfitting
   - Example: If your data contains only E-MTAB-2372 and E-GEOD-44763, then one would be used for training/validation and the other for testing

2. Organism/Donor Level:
   - Every human donor has a unique genome, lifestyle, and environmental exposures
   - Even inbred mice show biological variability
   - Data from the same donor should NEVER be split across train/val/test sets
   - All samples from a single donor must stay together in the same split

3. Batch Level:
   - Represents groups of samples processed together using the same protocol and personnel (even if they are from different donors)
   - If batch information is available, consider stratifying splits by batch
   - Be careful not to confound batches with biological variables of interest

4. Sample Level:
   - Samples with the same ID should never be split across datasets
   - Multiple measurements from the same sample must stay together in the same split

# ⚠️ CRITICAL REQUIREMENT: DATASET-LEVEL SPLITTING ⚠️
# You must implement a data splitting strategy that:
#
# 1. Identifies all unique dataset/study identifiers in the metadata
#
# 2. IF MULTIPLE DATASETS ARE PRESENT, separates datasets between test and train/val as follows:
#    - NEVER select just one dataset for testing
#    - Allocate at least 20% of total datasets to test set, when possible
#    - Select datasets that collectively provide adequate representation across:
#      * Age distribution (priority #1)
#      * Sex distribution (priority #2)
#      * Tissue types (priority #3)
#
# 3. Ensures NO overlap of datasets between test and train/val when multiple datasets exist
#
# 4. Provides SUFFICIENT test data (at least 20% of total samples, when possible)
#
# 5. For train/val splits, you do NOT need to split by dataset
#    - Train/val can come from the same datasets
#    - Use standard stratification approaches for train/val

For the epigenetic clock model:
- Age prediction is the primary goal, so ensure age distributions are similar across splits
- Consider tissue-type representation in each split, as methylation patterns vary by tissue. 
        - Even if it's not possible to split perfectly, try to ensure that the splits are as balanced as possible.
- Sex is a known confounder in methylation studies and should be balanced in each split
- Always prioritize higher-level sources of variance (study/dataset) over lower-level ones (donor/batch/sample) when making splitting decisions

Your output should be a detailed specification document that Team B can implement. It MUST include 
STEP-BY-STEP instructions for the engineer to follow!
The Principal Scientist will summarize the discussion and provide the final specification when the team reaches consensus.

Remember to:
- Build on each other's expertise
- Consider both biological and technical aspects
- Be specific about your requirements and justifications
- Ensure the splitting approach supports the goal of building an accurate epigenetic clock
- Consider practical implementation constraints
"""

SUBTASK_2_TEXT="""
# CURRENT WORKFLOW STEP: Task 3, Subtask 2 - Data Splitting Implementation

# TEAM COMPOSITION:
# - ML/Data Engineer: Implements the technical solutions in code based on the team's plans
# - Code Executor: Executes the code and returns the results.

As the ML/Data Engineer, you have received the data splitting specification from Team A. Your task is to:

# ⚠️ CRITICAL REQUIREMENT: DATASET-LEVEL SPLITTING ⚠️
# You must implement a data splitting strategy that:
#
# 1. Identifies all unique dataset/study identifiers in the metadata
#
# 2. IF MULTIPLE DATASETS ARE PRESENT, separates datasets between test and train/val as follows:
#    - NEVER select just one dataset for testing
#    - Allocate at least 20% of total datasets to test set, when possible
#    - Select datasets that collectively provide adequate representation across:
#      * Age distribution (priority #1)
#      * Sex distribution (priority #2)
#      * Tissue types (priority #3)
#
# 3. Ensures NO overlap of datasets between test and train/val when multiple datasets exist
#
# 4. Provides SUFFICIENT test data (at least 20% of total samples, when possible)
#
# 5. For train/val splits, you do NOT need to split by dataset
#    - Train/val can come from the same datasets
#    - Use standard stratification approaches for train/val

1. Review the specification document carefully and execute the following ordered steps:

   a) FIRST: Identify and analyze all unique dataset/study identifiers in the metadata:
      - Count samples per dataset
      - Analyze age/sex/tissue distribution within each dataset
      - If multiple datasets exist, determine which combinations would make a representative test set
   
   b) SECOND: If multiple datasets exist, strategically select datasets for the test set that:
      - Collectively contain at least 20% of the total samples, when possible
      - Represent the overall data distribution as closely as possible
      - Cover the age range adequately (most important)
      - If only one dataset exists, use standard random splitting techniques instead
   
   c) THIRD: Allocate remaining datasets to train/validation (or split the single dataset if only one exists)
      - Split these into train and validation using standard techniques
      - Stratify by age, sex, and tissue type
   
   d) FOURTH: Verify the representativeness of all splits
      - Compare distributions across train, val, and test
      - Ensure test set is sufficiently representative

2. Implement the requested data splitting using the provided tools:
   - Code runner with pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
   - Local file system access

3. Create the following data splits:
   - Training set
   - Validation set
   - Test set (containing samples from datasets completely separate from train/val)

4. Generate appropriate evaluation metrics and visualizations to demonstrate:
   - The distribution of datasets across the splits (MUST show which datasets went to which split)
   - Sample counts per dataset and per split
   - The distribution of key variables (age, sex, tissue) in each split
   - Statistical tests comparing distributions between splits
   - Any identified issues or concerns

5. Save the split datasets to the specified output files:
   - train.arrow, val.arrow, test.arrow (for the feature data)
   - train_meta.arrow, val_meta.arrow, test_meta.arrow (for the metadata)

6. Create a comprehensive report that includes:
   - Explicit explanation of your dataset selection strategy for test
   - Justification for why your test set will provide reliable evaluation
   - Confirmation that datasets were properly separated 
   - Sample counts and percentages for each split
   - Distribution analysis of key variables across splits
   - Code implementation details
   - Visualizations of all relevant distributions
   - Any limitations of your approach

# ⚠️ CRITICAL PLOT QUALITY REQUIREMENTS ⚠️
# You MUST verify all visualizations for quality and accuracy:
#
# 1. READABILITY: All labels, legends, and text must be clearly visible and appropriately sized
#
# 2. ACCURACY: All data categories must be correctly represented with appropriate colors and proportions
#
# 3. VERIFICATION: For EACH plot:
#    - Print a summary table showing counts and percentages that match what's in the plot
#    - Explicitly verify the numbers match what's visually represented
#    - If any category appears missing or incorrectly sized, recreate the plot
#
# 4. For gender/sex distribution plots specifically:
#    - Always create a cross-tabulation showing counts AND percentages by split
#    - Verify ALL genders are represented and proportionally correct
#    - Show exact counts before and after creating the plot
#
# 5. For tissue type or other categorical plots:
#    - Use horizontal orientation for better readability when many categories exist
#    - Ensure all category labels are readable
#    - Group small categories as "Other" if there are too many to display clearly

Your report should be:
- Well-documented with clear explanations
- Include all relevant code and outputs
- Highlight important patterns and insights
- Note any technical challenges encountered
- Confirm that all required output files have been created

# CRITICAL COMPLETION PROTOCOL
# To properly conclude your task:
# 1. Verify all output files exist and are readable using code - YOU MUST RUN A FINAL CODE BLOCK FOR THIS
# 2. Provide a concise summary of what was accomplished
# 3. Include a "Generated Files" section listing all output files with their purposes
# 4. Include a "Generated Plots" section listing all created visualizations
# 5. Follow your system prompt guidelines for properly finalizing your report


Remember to:
- Follow best practices in scientific computing
- Include appropriate error handling
- Make code reusable and maintainable
- Document all implementation choices
"""

SUBTASK_2_REVISION_TEXT="""
# CURRENT WORKFLOW STEP: Task 3, Subtask 2 - Data Splitting Implementation (REVISION)

# TEAM COMPOSITION:
# - ML/Data Engineer: Implements the technical solutions in code based on the team's plans
# - Code Executor: Executes the code and returns the results.

# IMPORTANT: This is ITERATION {iteration} of this subtask. You are expected to address the feedback provided by Team A in their review.

# ⚠️ CRITICAL REQUIREMENT: DATASET-LEVEL SPLITTING ⚠️
# If your previous implementation did not properly split by dataset/study:
#
# IF MULTIPLE DATASETS EXIST:
# 1. You MUST select MULTIPLE datasets for testing
#
# 2. Your test datasets should:
#    - Collectively contain at least 20% of total samples, when possible
#    - Represent the overall distribution of age, sex, and tissue
#    - Have NO overlap with train/validation datasets
#
# IF ONLY ONE DATASET EXISTS:
# 3. You should use appropriate stratification techniques
#
# 4. You MUST explicitly demonstrate the representativeness of your test set
#    with clear visualizations and statistics

As the ML/Data Engineer, you have received FEEDBACK on your previous implementation from Team A. Your task is to:

1. Review Team A's feedback carefully and acknowledge each point
2. Implement the requested revisions to the data splitting approach
3. Address all the issues and suggestions raised by the review team
4. Enhance your previous implementation based on their guidance
5. Create a revised comprehensive report that includes:
   - Response to each feedback point
   - Updated code implementation with proper dataset-level splitting
   - Clear tables showing sample counts per dataset and per split
   - Visualizations comparing distributions across splits
   - Statistical tests comparing distributions across splits
   - Confirmation of updated output files

# ⚠️ CRITICAL PLOT QUALITY REQUIREMENTS ⚠️
# You MUST verify all visualizations for quality and accuracy:
#
# 1. READABILITY: All labels, legends, and text must be clearly visible and appropriately sized
#
# 2. ACCURACY: All data categories must be correctly represented with appropriate colors and proportions
#
# 3. VERIFICATION: For EACH plot:
#    - Print a summary table showing counts and percentages that match what's in the plot
#    - Explicitly verify the numbers match what's visually represented
#    - If any category appears missing or incorrectly sized, recreate the plot
#
# 4. For gender/sex distribution plots specifically:
#    - Always create a cross-tabulation showing counts AND percentages by split
#    - Verify ALL genders are represented and proportionally correct
#    - Show exact counts before and after creating the plot
#
# 5. For tissue type or other categorical plots:
#    - Use horizontal orientation for better readability when many categories exist
#    - Ensure all category labels are readable
#    - Group small categories as "Other" if there are too many to display clearly
#
# 6. VISUAL INSPECTION: For every plot, ask yourself these questions:
#    - "Does this plot clearly show what I intend it to show?"
#    - "Are all categories represented and visible?"
#    - "Would someone unfamiliar with this data understand the visualization?"
#    - "Do the visual proportions match the numerical statistics?"
#    If the answer to ANY question is "no," you MUST revise the plot

Your revised report should:
- Begin with a clear acknowledgment of the feedback points you're addressing
- Explicitly show your dataset selection strategy for the test set
- Provide justification for why your test set is representative
- Include sample counts and percentages for each split
- Show distribution analysis of key variables across splits
- Note any ongoing challenges or limitations
- Confirm that all required output files have been created or updated

# CRITICAL COMPLETION PROTOCOL
# To properly conclude your task:
# 1. Verify all output files exist and are readable using code - YOU MUST RUN A FINAL CODE BLOCK FOR THIS
# 2. Provide a concise summary of what was accomplished
# 3. Include a "Generated Files" section listing all output files with their purposes
# 4. Include a "Generated Plots" section listing all created visualizations
# 5. Follow your system prompt guidelines for properly finalizing your report

Remember to:
- Maintain high-quality code with appropriate comments
- Continue following computational efficiency guidelines
- Ensure all visualizations are properly saved and documented
- Address ALL the feedback points from Team A
- Provide a complete, standalone report that incorporates the revisions

# Edit the file at experiments/altum_v1/03_split_data.py
fixed_text = SUBTASK_2_REVISION_TEXT.replace("{file}", "file")
task_text = fixed_text.format(iteration=iteration)

# Add this specific instruction about tabular statistics
task_text += f"\n\n⚠️ CRITICAL REPORT FORMAT REQUIREMENT ⚠️"
task_text += f"\n- Your FINAL report message MUST directly include all tabular statistics in plaintext format"
task_text += f"\n- Do NOT just print statistics during code execution and omit them from your final message"
task_text += f"\n- Copy-paste all important statistics tables directly into your final report (in markdown format) before saying the termination phrase / token."
task_text += f"\n- The critic can ONLY evaluate statistics that appear in your final message"
task_text += f"\n- Statistics printed during code execution but omitted from your final report will be IGNORED by the critic"
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

def get_task_workdir(stage, clean=False, workdir_suffix=None):
    """Get the task-specific working directory for engineer outputs.
    
    Args:
        stage: The stage number
        clean: Whether to clean the directory if it exists
        workdir_suffix: Optional suffix to add to the workdir name
        
    Returns:
        str: Path to the task-specific working directory
    """
    # Create base name like 'task_3_workdir' or 'task_3_custom_suffix'
    if workdir_suffix:
        workdir_name = f"task_{stage}_{workdir_suffix}"
    else:
        workdir_name = f"task_{stage}_workdir"
    
    # Ensure the task workdir exists
    os.makedirs(workdir_name, exist_ok=True)
    
    # Clean the directory if requested
    if clean:
        # Remove all files in the directory but keep the directory itself
        for item in os.listdir(workdir_name):
            item_path = os.path.join(workdir_name, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                    print(f"Cleaned: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Cleaned directory: {item_path}")
            except Exception as e:
                print(f"Error cleaning {item_path}: {e}")
        
        print(f"Directory cleaned: {workdir_name}")
        
    return workdir_name

async def run_subtask_1(task_env=None):
    """Run the first subtask: Team discussion to create data splitting specification."""
    # Get the workflow state
    state = get_workflow_state()
    stage = DATA_SPLIT_STAGE
    subtask = 1
    iteration = 1  # Subtask 1 only runs once
    
    # Update workflow state to indicate we're in DATA_SPLIT stage, subtask 1
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
        termination_condition=text_termination,
        max_turns=5  # Allow enough turns for thorough discussion
    )
    
    # Format the task with previous context
    formatted_task = await format_structured_task_prompt(stage, subtask, SUBTASK_1_TEXT, iteration)
    
    result = await Console(task_group.run_stream(task=formatted_task))
    
    # Save messages and summary with task description using structured approach
    await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], SUBTASK_1_TEXT)
    
    return result, task_env

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
    
    # Add plot quality checklist to the environment (for data splitting stage)
    if stage == DATA_SPLIT_STAGE:
        with open(os.path.join(output_dir, "plot_quality_checklist.txt"), "w") as f:
            f.write(PLOT_QUALITY_CHECKLIST)
    
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

# Function to prepare plot search instructions for the engineer
def add_plot_instructions(task_text, output_dir):
    """Add instructions on how to ensure plot files can be found by the critic."""
    plot_instructions = f"""
IMPORTANT VISUALIZATION NAMING REQUIREMENTS:
- Save all plots to the '{output_dir}' directory
- In your final report, include a clearly labeled "Generated Plots:" section
- List EACH visualization filename with the .png extension (e.g., age_distribution.png)
- The critic will search for these EXACT filenames - do not change names after listing them
- For each filename listed, include a brief description of what the plot shows
- Example format:
  Generated Plots:
  1. age_distribution.png - Shows age distribution across splits
  2. gender_distribution.png - Shows gender balance in each split
  3. tissue_distribution.png - Shows tissue type representation
- Verify all plot files exist and are readable before completing your report
"""
    return task_text + plot_instructions

# Function to add engineer code execution guardrails
def add_code_execution_guardrails(task_text):
    """Add explicit guardrails against incorrect code block formatting."""
    guardrails = """
⚠️ CRITICAL CODE EXECUTION REQUIREMENTS ⚠️

1. ONLY USE EXECUTABLE CODE BLOCKS: 
   - ALWAYS use ```bash code blocks for executable code
   - NEVER use ```plaintext, ```json, or any other non-executable format

2. VALID CODE EXECUTOR FORMAT:
   ```bash
   #!/bin/bash
   eval "$(conda shell.bash hook)"
   conda activate sklearn-env
   PYTHONFAULTHANDLER=1 python - <<END
   import pandas as pd
   import matplotlib.pyplot as plt
   # Your actual Python code here
   END
   ```

3. TOOL USAGE RULES:
   - DO NOT use multi_tool_use.parallel() or any similar patterns
   - Use tools ONE AT A TIME in separate messages
   - Always wait for each tool's response before using another tool

4. FOR ANALYZE_PLOT AND SEARCH_DIRECTORY:
   - Use these tools directly in your regular message format
   - Example: analyze_plot("task_3_workdir/age_distribution.png")
   - NOT in a code block

5. INVALID FORMATS (DO NOT USE):
   ```plaintext
   multi_tool_use.parallel({
     "tool_uses": [...]
   })
   ```
   OR
   ```json
   {
     "tool": "analyze_plot",
     "parameters": {...}
   }
   ```

FAILURE TO FOLLOW THESE GUIDELINES WILL CAUSE CODE EXECUTION TO FAIL.
"""
    return task_text + guardrails

async def run_subtask_2(iteration=1, task_env=None, retry_count=0):
    """Run the second subtask: Engineer implementing the data splitting specification.
    
    Args:
        iteration: The iteration number for this subtask
        task_env: Optional task environment including workdir
        retry_count: Current retry attempt (0 = first attempt)
        
    Returns:
        Result object or None if all retries failed
    """
    # Update workflow state
    stage = DATA_SPLIT_STAGE
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
            sleep(1)
        
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        
        # Make sure the directory search tool is available 
        from altum_v1.tools import search_directory
        if 'search_directory' not in available_tools:
            available_tools['search_directory'] = search_directory
        
        # Load the model client for all agents
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        
        # 1. Create the engineer agent with all tools
        which_agents = ['engineer']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)
        engineer_agent = list(agents.values())[0]
        
        # Note: System message should be configured in agents.yaml, not modified here
        
        # 2. Create the code executor agent
        code_executor = DockerCommandLineCodeExecutor(
            image='agenv:latest',
            work_dir=task_env["workdir"],  # Use main working directory for Docker access
            timeout=600
        )
        await code_executor.start()
        code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
        
        # 3. Create the critic agent with ReAct style prompt
        critic_agent = AssistantAgent(
            name="data_science_critic",
            system_message=CRITIC_SYSTEM_PROMPT,
            model_client=model_client,
            tools=[tool for tool in available_tools.values()],
            model_client_stream=True,
            reflect_on_tool_use=True
        )
        
        # 4. Create the summarizer agent
        summarizer_agent = AssistantAgent(
            name="technical_summarizer",
            system_message=SUMMARIZER_SYSTEM_PROMPT,
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
        
        # Choose the appropriate task text based on iteration
        if iteration == 1:
            task_text = SUBTASK_2_TEXT
        else:
            # Preemptively replace any {file} placeholder before formatting
            fixed_text = SUBTASK_2_REVISION_TEXT.replace("{file}", "file")
            task_text = fixed_text.format(iteration=iteration)
        
        # Append directory information to the task text
        task_text += f"\n\nIMPORTANT FILE ORGANIZATION INSTRUCTIONS:"
        task_text += f"\n- Your code runs in the main project directory where data files are located"
        task_text += f"\n- SAVE ALL OUTPUT FILES to the '{task_env['output_dir']}' directory"
        task_text += f"\n- This includes plots, intermediate data files, and any other outputs"
        task_text += f"\n- Example: plt.savefig('{task_env['output_dir']}/my_plot.png')"
        task_text += f"\n- The split data files should be saved as train.arrow, val.arrow, test.arrow and train_meta.arrow, val_meta.arrow, test_meta.arrow"
        
        # Add data format guardrail - new addition
        task_text += f"\n\n⚠️ CRITICAL DATA FORMAT REQUIREMENT ⚠️"
        task_text += f"\n- ALWAYS save dataframes ONLY in arrow/feather format using .to_feather() or similar methods"
        task_text += f"\n- NEVER save data to CSV, TSV, or any other text-based format as these are highly inefficient"
        task_text += f"\n- Example: df.to_feather('{task_env['output_dir']}/data.arrow')"
        task_text += f"\n- For any intermediate files, also use arrow/feather format"
        task_text += f"\n- If you need to export small amounts of summary statistics, use JSON or pickle, but not CSV"
        
        # Add specific plot file naming instructions
        task_text = add_plot_instructions(task_text, task_env['output_dir'])
        
        # Add code execution guardrails
        task_text = add_code_execution_guardrails(task_text)
        
        # Add information about plot quality requirement
        task_text += f"\n\nANALYSIS QUALITY VERIFICATION REQUIREMENT:"
        task_text += f"\n- A plot quality checklist has been saved to '{task_env['output_dir']}/plot_quality_checklist.txt'"
        task_text += f"\n- For your analysis, you MUST:"
        task_text += f"\n  1. Provide detailed TABULAR STATISTICS for all key metrics within the text of your analysis report"
        task_text += f"\n  2. Include cross-tabulations showing counts AND percentages for categorical variables"
        task_text += f"\n  3. Use statistical tests to verify the representativeness of splits"
        task_text += f"\n  4. Present tables BEFORE visualizations as your primary evidence"
        task_text += f"\n  5. Use visualizations to support and enhance the tabular statistics"
        task_text += f"\n- For categorical variables (especially gender/sex and tissue):"
        task_text += f"\n  1. Always include detailed cross-tabulation tables showing counts AND percentages by group"
        task_text += f"\n  2. Format tables for maximum readability (e.g., using pandas styling or clear formatting)"
        task_text += f"\n  3. Include row and column totals in your tables where appropriate"
        
        # Add information about the critic's ReAct process and directory search tool
        task_text += f"\n\nREVIEW PROCESS AND TOOLS:"
        task_text += f"\n- After you complete your implementation, a data science critic will review your work"
        task_text += f"\n- The critic will evaluate both your methodology and the statistical evidence you provide"
        task_text += f"\n- The critic strongly prefers tabular summary statistics over visualizations"
        task_text += f"\n- Address ALL feedback from the critic by improving your analysis and evidence"
        task_text += f"\n- The subtask will only complete when the critic is satisfied with your methodology and evidence"
        
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
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
        # Save the summarized report for subtask 3 to use
        if result.chat_message and isinstance(result.chat_message, TextMessage):
            summary_file_path = os.path.join(task_env['output_dir'], f"implementation_summary_iteration_{iteration}.txt")
            with open(summary_file_path, "w") as f:
                f.write(result.chat_message.content)
            print(f"Saved implementation summary to {summary_file_path}")
        
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

async def main(args=None):
    """Run the complete data splitting workflow."""
    
    # Default values
    restart_stage = DATA_SPLIT_STAGE
    is_restart = False
    resume_checkpoint = None
    task_env = None
    
    print("\n" + "="*80)
    print("⚠️ CRITICAL DATA SPLITTING REQUIREMENT ⚠️")
    print("IF MULTIPLE DATASETS EXIST:")
    print("  - The test set MUST contain samples from multiple datasets not in train/val")
    print("  - Test set should be at least 20% of data, when possible")
    print("  - Test and train/val sets MUST have completely separate datasets")
    print("IF ONLY ONE DATASET EXISTS:")
    print("  - Use appropriate stratification techniques for splitting")
    print("="*80 + "\n")
    
    if not args:
        # If running from command line, get workflow options
        action = await prompt_for_workflow_action(DATA_SPLIT_STAGE)
        
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
                            
                            # Mark stage as completed since we're skipping subtask 3
                            print(f"Data splitting completed after iteration {current_iteration}")
                            mark_stage_completed(DATA_SPLIT_STAGE)
                            save_workflow_checkpoint(MODEL_TRAINING_STAGE, label="Ready for Model Training")
                            return
                        
                        # If we get here, we're done with the resume-specific logic
                        return
        else:  # New workflow
            print("Starting new data splitting workflow...")
            # Set workflow state to DATA_SPLIT stage
            update_workflow_state(DATA_SPLIT_STAGE)
            restart_stage = DATA_SPLIT_STAGE
            is_restart = True
    else:
        # If called programmatically with args
        restart_stage = args.get("restart_stage", DATA_SPLIT_STAGE)
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
    save_workflow_checkpoint(DATA_SPLIT_STAGE, label="Data Splitting Start")
    
    # Start the workflow from the appropriate point
    if restart_stage == DATA_SPLIT_STAGE:
        # Check if EDA stage is completed
        if not is_stage_completed(EDA_STAGE):
            print("Warning: EDA stage (Stage 2) has not been completed.")
            proceed = input("Do you want to proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Exiting. Please run Stage 2 first.")
                return
    
        # Run subtask 1: Team discussion to create data splitting specification
        result1, task_env = await run_subtask_1(task_env)
        if not result1:
            print("Subtask 1 failed to complete")
            return
        
        # Save checkpoint after subtask 1
        save_workflow_checkpoint(DATA_SPLIT_STAGE, 1, 1, "Data Splitting Specification Completed")
    
    # Run subtask 2: Engineer implementing the data splitting
    iteration = 1
    result2 = await run_subtask_2(iteration, task_env)
    if not result2:
        print("Subtask 2 failed to complete after multiple retries")
        return
    
    # Save checkpoint after subtask 2
    save_workflow_checkpoint(DATA_SPLIT_STAGE, 2, iteration, f"Data Splitting Implementation (Iteration {iteration})")
    
    # Mark stage as completed (skipping subtask 3)
    print(f"Data splitting completed after iteration {iteration}")
    mark_stage_completed(DATA_SPLIT_STAGE)
    save_workflow_checkpoint(MODEL_TRAINING_STAGE, label="Ready for Model Training")
    
    # Final cleanup of temporary files
    cleanup_temp_files()

if __name__ == "__main__":
    # Add command-line arguments
    parser = argparse.ArgumentParser(description="Run the data splitting workflow with checkpoint/resume options")
    parser.add_argument("--restart", action="store_true", help="Restart from beginning of data splitting stage")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--force", action="store_true", help="Force restart without prompting")
    args_parsed = parser.parse_args()
    
    if args_parsed.restart and args_parsed.resume:
        print("Error: Cannot specify both --restart and --resume")
        sys.exit(1)
    
    if args_parsed.restart:
        # Restart from beginning of data splitting
        asyncio.run(clear_workflow_state(DATA_SPLIT_STAGE))
        asyncio.run(main({"restart_stage": DATA_SPLIT_STAGE, "clear_state": True}))
    elif args_parsed.resume:
        # Resume from latest checkpoint (interactive)
        asyncio.run(main())
    else:
        # Interactive mode
        asyncio.run(main())






