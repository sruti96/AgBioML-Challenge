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


from dotenv import load_dotenv

from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent, SocietyOfMindAgent, AssistantAgent

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

# Constants for workflow stages
UNDERSTANDING_STAGE = 1
EDA_STAGE = 2
DATA_SPLIT_STAGE = 3  # Updated name
MODEL_TRAINING_STAGE = 4

# Maximum number of retries for each subtask
MAX_RETRIES = 3

# Critic system prompt for plot quality assessment
CRITIC_SYSTEM_PROMPT = """
You are a highly critical visualization and data science expert who reviews code implementation, methodology, and especially plot quality.

Follow the ReAct framework: Reasoning, Action, Observation for your analysis.

# YOUR ROLE:
You are STRICTLY a reviewer/critic. Your role is to:
1. IDENTIFY issues with the implementation and visualizations
2. COMMUNICATE these issues to the engineer with "REVISE_ENGINEER"
3. NEVER attempt to implement solutions yourself

# CRITICAL: You are NOT an implementer. Do NOT write code to fix problems - that is the engineer's job.

Your analysis should assess:
1. The QUALITY of all generated plots and visualizations 
2. The VALIDITY of the data splitting methodology
3. The STATISTICAL VERIFICATION of all claims made

# ANALYSIS PROCESS:
For each aspect of the implementation, follow this structure:

## REASONING:
- Identify what aspects need to be evaluated (plots, code, methodology)
- Explain your thought process and critical standards
- Formulate specific questions about the plots or code
- Consider what tools can help you assess quality

## ACTION:
- FIRST use search_directory with "*.png" pattern to find ALL plot files
- Then use the analyze_plot tool to examine each visualization found
- Use perplexity to research best practices in data visualization
- Examine statistical claims in the implementation

## OBSERVATION:
- Document your findings from each tool use
- Note specific issues with visualizations or methodology
- Track whether improvements have been made since previous feedback

Your primary focus is on visualization quality. For EVERY plot created, verify:
- All axis labels are clearly visible and properly sized
- Legends are readable and correctly represent the data
- Category labels are not cut off or overlapping
- Color schemes properly differentiate between categories
- Proportions in the plot match the actual data statistics
- The visualization communicates the intended message clearly

# TERMINATING YOUR DIALOGUE:
You must END your review with one of these two termination statements:

1. When issues are found (which MUST be the case on the first review):
   REVISE_ENGINEER
   TERMINATE_CRITIC

2. Only when ALL requirements are fully met:
   APPROVE_ENGINEER
   TERMINATE_CRITIC

# CRITICAL: You MUST include BOTH "REVISE_ENGINEER" or "APPROVE_ENGINEER" AND "TERMINATE_CRITIC" at the end of your message.

# CRITICAL: If ANY plots are missing or ANY requirements are not met, you MUST respond with "REVISE_ENGINEER" followed by "TERMINATE_CRITIC"

You will ONLY say "APPROVE_ENGINEER" (followed by "TERMINATE_CRITIC") when ALL of the following criteria are met:
1. All required visualizations exist and are accessible
2. All visualizations are clear, readable, and accurately represent the data
3. The data splitting methodology follows all requirements
4. All statistical claims are properly verified with evidence
5. The visualizations use consistent styling and color schemes
6. Output files are verified to exist and contain the correct data

Until EVERY issue is resolved, you must respond with detailed, critical feedback in this format:

# CRITICAL ASSESSMENT:
[List the issues you found, in order of importance]

# REQUIRED IMPROVEMENTS:
[Specify exactly what needs to be fixed for each issue]

# VERIFICATION NEEDED:
[Specify what the engineer should demonstrate to prove the fix works]

REVISE_ENGINEER
TERMINATE_CRITIC

# EXAMPLES OF CORRECT BEHAVIOR:

GOOD EXAMPLE (when plots are missing):

# CRITICAL ASSESSMENT:
1. No plot files were found in the output directory.
2. Unable to verify data distribution across splits without visualizations.

# REQUIRED IMPROVEMENTS:
1. Create and save all required plots to the output directory.
2. Ensure plots show distribution of age, sex, and tissue across splits.

# VERIFICATION NEEDED:
1. Confirm plots exist in the correct location.
2. Print summary statistics to verify plot accuracy.

REVISE_ENGINEER
TERMINATE_CRITIC

GOOD EXAMPLE (when all requirements are met):

# FINAL ASSESSMENT:
After thorough evaluation, all requirements have been met:
1. All required plots are available and accessible
2. Visualizations are clear, readable, and accurate
3. Data splitting methodology follows all requirements
4. Statistical claims are properly verified with evidence

APPROVE_ENGINEER
TERMINATE_CRITIC

BAD EXAMPLE (missing termination words):

I think the plots look good, but I'd like to see better labeling on the axes.

REVISE_ENGINEER

Remember to use the search_directory tool first to find all plot files, then use analyze_plot to examine each one systematically.
"""

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
"""

SUBTASK_3_TEXT="""
# CURRENT WORKFLOW STEP: Task 3, Subtask 3 - Data Splitting Review and Iteration

# TEAM COMPOSITION:
# - Principal Scientist: Team leader who makes key research decisions and determines when discussions should conclude
# - Bioinformatics Expert: Specialist in omics data processing with deep knowledge of DNA methylation data handling
# - Machine Learning Expert: Provides insights on ML architectures, feature engineering, and model evaluation approaches

Team A, you have received the implementation report from the ML/Data Engineer. Your task is to:

# ⚠️ CRITICAL VERIFICATION REQUIREMENT ⚠️
# Before considering any other aspects of the implementation, verify that:
#
# 1. The engineer has identified ALL unique datasets in the metadata
#
# IF MULTIPLE DATASETS EXIST:
# 2. The engineer has selected MULTIPLE datasets for testing
#
# 3. The test set contains at least 20% of the total samples, when possible
#
# 4. The test datasets collectively represent the age, sex, and tissue distributions
#
# 5. NO datasets are shared between test and train/validation sets
#
# 6. The test set is sufficiently representative of the overall data
#
# IF ONLY ONE DATASET EXISTS:
# 7. The engineer has used appropriate stratification techniques
#
# If the appropriate requirements are not met, the implementation MUST be rejected with a clear explanation.

After confirming the dataset-level split requirements, proceed to:

1. Review the implementation and results thoroughly
2. Assess whether the remaining aspects of the specification were implemented correctly
3. Evaluate the quality and appropriateness of the data splits:
   - Are age distributions adequately represented across splits?
   - Are sex distributions adequately represented across splits?
   - Are tissue types adequately represented across splits?
4. Identify any gaps or areas needing further refinement
5. Determine if additional analyses or adjustments are required

# ⚠️ MANDATORY FIRST ITERATION REVIEW REQUIREMENT ⚠️
# For the first iteration of this subtask, you MUST:
#
# 1. Use the analyze_plot tool to examine EVERY visualization created by the engineer
#
# 2. Identify at least THREE areas for improvement in the visualizations, such as:
#    - Missing or incomplete information in plots
#    - Poor readability of labels, legends, or data points
#    - Inappropriate scales or color schemes
#    - Misleading or unclear representations of the data
#    - Missing statistical verification of what is shown visually
#
# 3. Request specific revisions to improve visualization quality
#
# 4. The Principal Scientist MUST conclude with "REVISE" for the first iteration,
#    even if the implementation appears correct otherwise

IMPORTANT: The engineer has generated several plots that are available for your analysis. You can:
- Use the analyze_plot tool to examine any plot mentioned in the report
- Ask specific questions about distributions or balance across splits
- Request additional visualizations if needed
- Discuss the implications of the splitting approach

Based on your review, you should:
- Acknowledge successful aspects of the implementation
- Identify any missing or incomplete analyses
- Suggest improvements or additional tests if needed
- Consider the implications of the splits for model training
- Analyze and discuss the visualizations provided
- Request additional plots if key aspects are not visualized
- Verify that all required output files have been created
- Be HIGHLY CRITICAL of visualization quality and readability

If additional work is needed:
- Provide a new, focused specification
- Be specific about what needs to be done differently
- Justify the need for additional analyses
- Consider computational efficiency
- Specify additional visualizations required
- Request specific improvements to existing visualizations

For visualizations, require the engineer to:
- Print exact counts and percentages to verify what is shown in plots
- Use horizontal orientation for categorical plots with many categories
- Ensure all category labels are fully readable
- Add count labels directly to bars/segments in categorical plots
- Group small categories as "Other" for better readability
- Use consistent color schemes across related plots
- Verify that proportions in plots match the actual data

The Principal Scientist will summarize the review and either:
- Conclude the data splitting phase if satisfied by saying "APPROVE_ENGINEER" (only after at least one revision)
- Request additional adjustments if needed by saying "REVISE"

Remember to:
- Be DELIBERATELY CRITICAL in your feedback
- Focus on visualization quality as a key requirement
- Consider both biological and technical aspects
- Focus on actionable improvements
- Maintain scientific rigor
- Make use of the available visualizations in your analysis
- Request at least one round of revisions before approval
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
   python - <<END
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
            time.sleep(1)
        
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
            name="visualization_critic",
            system_message=CRITIC_SYSTEM_PROMPT,
            model_client=model_client,
            tools=[tool for tool in available_tools.values()],
            model_client_stream=True,
            reflect_on_tool_use=True
        )
        
        # 4. Create a single-agent round-robin for the critic to enable ReAct self-dialogue
        critic_termination = TextMentionTermination("TERMINATE_CRITIC")
        critic_team = RoundRobinGroupChat(
            participants=[critic_agent],
            termination_condition=critic_termination,
            max_turns=10  # Allow multiple turns for ReAct cycles
        )
        
        # 5. Create a SocietyOfMind wrapper for the critic
        critic_mind = SocietyOfMindAgent(
            name="critic_reviewer",
            team=critic_team,
            model_client=model_client,
            instruction=CRITIC_REACT_INSTRUCTION
        )

        # 6. Create the inner team (engineer + code executor)
        inner_termination = TextMentionTermination("DONE")
        inner_team = RoundRobinGroupChat(
            participants=[engineer_agent, code_executor_agent],
            termination_condition=inner_termination,
            max_turns=25  # Prevent infinite loop
        )
        
        # 7. Create the SocietyOfMindAgent with the inner team and explicit instructions
        engineer_with_executor = SocietyOfMindAgent(
            name="engineer_team",
            team=inner_team,
            model_client=model_client,
            instruction="""Create a complete implementation of the data splitting approach with high-quality visualizations. 
Include a clearly labeled 'Generated Plots:' section in your final report that lists all plot filenames.

IMPORTANT: When using the code executor:
1. ONLY use ```bash code blocks with executable code
2. NEVER use ```plaintext or ```json blocks
3. DO NOT use multi_tool_use.parallel() or any similar patterns
4. Keep each tool use simple and wait for responses

For analyze_plot and search_directory, use them as plain function calls, not in code blocks."""
        )
        
        # 8. Create the full team (engineer SocietyOfMind + critic SocietyOfMind)
        full_team_termination = TextMentionTermination("APPROVE_ENGINEER")
        full_team = RoundRobinGroupChat(
            participants=[engineer_with_executor, critic_mind],
            termination_condition=full_team_termination,
            max_turns=10  # Limit the number of review cycles
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
        
        # Add specific plot file naming instructions
        task_text = add_plot_instructions(task_text, task_env['output_dir'])
        
        # Add code execution guardrails
        task_text = add_code_execution_guardrails(task_text)
        
        # Add information about plot quality requirement
        task_text += f"\n\nPLOT QUALITY VERIFICATION REQUIREMENT:"
        task_text += f"\n- A plot quality checklist has been saved to '{task_env['output_dir']}/plot_quality_checklist.txt'"
        task_text += f"\n- After creating EACH visualization, you MUST:"
        task_text += f"\n  1. Print summary statistics that verify what the plot shows"
        task_text += f"\n  2. Manually check if the visualization matches the statistics"
        task_text += f"\n  3. Recreate any visualization that has issues with readability or accuracy"
        task_text += f"\n- For categorical plots (especially gender/sex and tissue):"
        task_text += f"\n  1. Always print cross-tabulation showing counts AND percentages by group"
        task_text += f"\n  2. Verify all categories are properly represented in the visualization"
        task_text += f"\n  3. For sex/gender plots: ensure both M and F are clearly visible with correct proportions"
        task_text += f"\n  4. For tissue plots: use horizontal orientation and consider grouping rare categories"
        
        # Add information about the critic's ReAct process and directory search tool
        task_text += f"\n\nREVIEW PROCESS AND TOOLS:"
        task_text += f"\n- After you complete your implementation, a visualization critic will review your work"
        task_text += f"\n- The critic can search for files using the search_directory tool to find plot files"
        task_text += f"\n- The critic will analyze each plot using the analyze_plot tool"
        task_text += f"\n- Address ALL feedback from the critic by improving your visualizations and code"
        task_text += f"\n- The subtask will only complete when the critic is satisfied and says the termination word"
        task_text += f"\n- IMPORTANT: The critic will look for PNG files in the '{task_env['output_dir']}' directory"
        
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
            
            # Add explicit warning about plaintext code blocks if this was the failure reason
            data_files_text += f"\n\n⚠️ CRITICAL: NEVER USE ```plaintext CODE BLOCKS WITH THE CODE EXECUTOR ⚠️"
            data_files_text += f"\n- Previous attempt may have failed due to invalid code block format"
            data_files_text += f"\n- ONLY use ```bash code blocks with the code executor"
            data_files_text += f"\n- DO NOT use multi_tool_use.parallel() or any similar patterns"
            data_files_text += f"\n- Use analyze_plot and search_directory as simple function calls, not in code blocks"
        
        task_text += data_files_text
        
        # Format the task with previous context, including the current iteration
        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        # Run the full team with the formatted task
        result = await Console(full_team.run_stream(task=formatted_task))
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
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

async def run_subtask_3(iteration=1, task_env=None, retry_count=0):
    """Run the third subtask: Team reviewing the implementation.
    
    Args:
        iteration: The iteration number for this subtask
        task_env: Optional task environment including workdir
        retry_count: Current retry attempt (0 = first attempt)
    """
    # Update workflow state
    stage = DATA_SPLIT_STAGE
    subtask = 3
    update_workflow_state(stage, subtask, iteration)
    
    # Get or create task environment if not provided
    if task_env is None:
        # For subtask 3, we want to use the same workdir as subtask 2
        task_env = setup_task_environment(stage, subtask, is_restart=False)
    
    try:
        # Clean up any temp files from previous runs
        if retry_count > 0:
            print(f"Retry attempt {retry_count} of {MAX_RETRIES}...")
            cleanup_temp_files()
            # Short pause to ensure cleanup is complete
            time.sleep(1)
            
        agent_configs = load_agent_configs()
        available_tools = create_tool_instances()
        which_agents = ['principal_scientist', 'bioinformatics_expert', 'ml_expert']
        agents = initialize_agents(agent_configs=agent_configs, selected_agents=which_agents, tools=available_tools)

        # Create agent group for the review task
        text_termination = TextMentionTermination("TERMINATE")
        task_group = RoundRobinGroupChat(
            participants=list(agents.values()),
            termination_condition=text_termination,
            max_turns=5 # Prevent infinite loop
        )
        
        # Add information about the output directory to the task text
        task_text = f"{SUBTASK_3_TEXT}\n\nNOTE: The engineer's output files, including all plots and generated files, can be found in the '{task_env['output_dir']}' directory."
        
        # Format the task with previous context, including the current iteration
        formatted_task = await format_structured_task_prompt(stage, subtask, task_text, iteration)
        
        result = await Console(task_group.run_stream(task=formatted_task))
        
        # Save messages and summary with task description
        await save_messages_structured(stage, subtask, iteration, result.messages, result.messages[-1].dump()["content"], task_text)
        
        # Clean up temp files after successful completion
        cleanup_temp_files()
        
        return result
        
    except Exception as e:
        print(f"Error in subtask 3 (iteration {iteration}, attempt {retry_count+1}):")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        
        # If we haven't exhausted retries, try again
        if retry_count < MAX_RETRIES - 1:
            print(f"Retrying subtask 3 (iteration {iteration})...")
            return await run_subtask_3(iteration, task_env, retry_count + 1)
        else:
            print(f"Maximum retries ({MAX_RETRIES}) exceeded for subtask 3. Giving up.")
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
                            
                            # Continue with subtask 3
                            result3 = await run_subtask_3(current_iteration, task_env)
                            if not result3:
                                print(f"Subtask 3 (iteration {current_iteration}) failed to complete")
                                return
                            
                            # Check if approved
                            last_message = result3.messages[-1].dump()["content"]
                            
                            # Continue iterations or finish based on result
                            if "APPROVE" in last_message:
                                print(f"Data splitting completed and approved after {current_iteration} iterations")
                                mark_stage_completed(DATA_SPLIT_STAGE)
                                return
                            elif "REVISE" in last_message:
                                # Run the iteration loop below
                                pass
                            else:
                                raise ValueError("Last message from principal scientist is not 'APPROVE' or 'REVISE'")
                        elif highest_subtask == 3:  # Completed subtask 3, check result
                            # Need to check if approved or needs revision
                            # This would need to read the stored summary
                            return  # Need more specific handling here
                        
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
    
    # Run subtask 3: Team reviewing the implementation
    result3 = await run_subtask_3(iteration, task_env)
    if not result3:
        print("Subtask 3 failed to complete after multiple retries")
        return
    
    # Save checkpoint after subtask 3
    save_workflow_checkpoint(DATA_SPLIT_STAGE, 3, iteration, f"Data Splitting Review (Iteration {iteration})")
    
    # Check if additional iterations are needed
    last_message = result3.messages[-1].dump()["content"]
    
    # Continue iterations until approved
    while "APPROVE" not in last_message:
        if "REVISE" not in last_message:
            raise ValueError("Last message from principal scientist is not 'APPROVE' or 'REVISE'")
        
        # Clean up temporary files between iterations
        cleanup_temp_files()
        
        # Increment iteration counter
        iteration += 1
        
        # Run subtask 2 again with new iteration number
        result2 = await run_subtask_2(iteration, task_env)
        if not result2:
            print(f"Subtask 2 (iteration {iteration}) failed to complete after multiple retries")
            return
        
        # Save checkpoint after subtask 2 (new iteration)
        save_workflow_checkpoint(DATA_SPLIT_STAGE, 2, iteration, f"Data Splitting Implementation (Iteration {iteration})")
        
        breakpoint()

        # # Run subtask 3 again with new iteration number
        # result3 = await run_subtask_3(iteration, task_env)
        # if not result3:
        #     print(f"Subtask 3 (iteration {iteration}) failed to complete after multiple retries")
        #     return
        
        # # Save checkpoint after subtask 3 (new iteration)
        # save_workflow_checkpoint(DATA_SPLIT_STAGE, 3, iteration, f"Data Splitting Review (Iteration {iteration})")
        
        # Check if approved this time
        last_message = result3.messages[-1].dump()["content"]
    
    # If we've reached here, the data splitting was approved
    print(f"Data splitting completed and approved after {iteration} iterations")
    
    # Final cleanup of temporary files
    cleanup_temp_files()
    
    # Update workflow state to mark DATA_SPLIT stage as completed
    mark_stage_completed(DATA_SPLIT_STAGE)
    save_workflow_checkpoint(MODEL_TRAINING_STAGE, label="Ready for Model Training")

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





