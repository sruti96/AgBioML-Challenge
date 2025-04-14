"""Custom agent implementations for the Altum v1 workflow."""

from typing import Sequence
import time

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console


# Function to estimate the number of tokens in a list of messages (moved from 03_split_data.py)
def estimate_tokens(messages):
    """Estimate the number of tokens in a list of messages."""
    # Get the total number of words in the messages
    total_words = sum(len(message.content.split()) for message in messages)
    # Get the total number of tokens in the messages (approximation)
    total_tokens = total_words * 3
    return total_tokens


class EngineerSociety(BaseChatAgent):
    """A custom agent that manages the interaction between an engineer team and a critic team.
    
    This replaces the previous SocietyOfMindAgent implementation with a more direct approach
    that cycles between the engineer team and the critic team until the critic approves.
    """
    def __init__(self, name: str, engineer_team: RoundRobinGroupChat, critic_team: RoundRobinGroupChat, 
                 critic_approve_token: str, engineer_terminate_token: str, critic_terminate_token: str, 
                 critic_revise_token: str, summarizer_agent=None, original_task=None, output_dir=".") -> None:
        super().__init__(name, description="An agent that performs implementation with critical feedback.")
        self._engineer_team = engineer_team
        self._critic_team = critic_team
        self._engineer_terminate_token = engineer_terminate_token
        self._critic_terminate_token = critic_terminate_token
        self._critic_approve_token = critic_approve_token
        self._critic_revise_token = critic_revise_token
        self._summarizer_agent = summarizer_agent
        self._original_task = original_task
        self._output_dir = output_dir
        self.messages_to_summarize = []  # Track all engineer messages

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages through the engineer team and critic team with a single round of review.
        
        The flow is:
        1. Engineer team (engineer + executor) writes and runs code
        2. Critic team reviews the results
        3. Result is summarized and returned regardless of critic approval
        """
        NUM_LAST_MESSAGES = 12
        original_messages = messages
        print(f"TOKEN ESTIMATE: engineer society: {estimate_tokens(messages)}")
        print(f"NUM MESSAGES: {len(messages)}")
        time.sleep(5)
        
        # Add instruction for the engineer to save files in the correct directory
        engineer_directory_instruction = TextMessage(
            content=f"""IMPORTANT FILE PATH INSTRUCTIONS:

ALL output files (plots, data, etc.) MUST be saved in this exact directory:
{self._output_dir}

Examples of correct file paths:
- plt.savefig('{self._output_dir}/histogram.png')
- df.to_csv('{self._output_dir}/results.csv')
- np.save('{self._output_dir}/array_data.npy')

Do NOT save files to the current directory or any other location. Always use '{self._output_dir}/' as the path prefix.
""",
            source="User"
        )
        
        # Add engineering heuristics and best practices
        engineering_heuristics = TextMessage(
            content="""ENGINEERING BEST PRACTICES AND TROUBLESHOOTING HEURISTICS:

When implementing your solution, follow these heuristics:

1. DATA UNDERSTANDING:
   - Always start by exploring and understanding the data structure (column names, data types, missing values)
   - Print shapes, descriptive statistics, and a few sample rows first
   - Check for missing data, outliers, or unusual distributions before proceeding

2. TROUBLESHOOTING APPROACH:
   - When you encounter an error, simplify your code to isolate the problem
   - Test individual components separately before combining them
   - Print intermediate results to verify each step works as expected
   - When debugging, start with the simplest possible version of your code

3. DEVELOPMENT STRATEGY:
   - Start small with atomic, focused steps that do one thing well
   - Test each component separately before combining them
   - Build up complexity incrementally, verifying at each step
   - Use intermediate data files to break complex processes into manageable stages

4. PERFORMANCE AND QUALITY:
   - Use sampling for initial testing when working with large datasets
   - Monitor memory usage and optimize for large data processing
   - Create clear, informative visualizations with proper labels and titles
   - Add useful comments explaining WHY, not just WHAT your code does

5. DATA SPLITTING BEST PRACTICES:
   - Always maintain stratification for important variables when splitting
   - Check the distributions in your train/test splits to ensure they're representative
   - Verify there's no data leakage between splits
   - Use k-fold cross-validation when appropriate to ensure stable results

6. OUTPUT VALIDATION:
   - Generate summary statistics for each data split and compare them
   - Create plots showing distributions across splits to visually confirm balance
   - Use statistical tests to verify similarity between splits
   - Create clear tables showing counts and percentages of key variables across splits

Remember to check your results at each step and build up complexity gradually.
""",
            source="User"
        )
        
        # Add the instruction to the messages
        engineer_messages_with_path = list(messages) + [engineer_directory_instruction]
        
        # Add heuristics to messages
        engineer_messages_with_path.append(engineering_heuristics)
        
        # Run the engineer team with the given messages
        result_engineer = await Console(self._engineer_team.run_stream(task=engineer_messages_with_path, cancellation_token=cancellation_token))
        
        engineer_messages = result_engineer.messages
        engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
        engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
        print(f"TOKEN ESTIMATE: engineer team: {estimate_tokens(engineer_messages)}")
        print(f"NUM MESSAGES: {len(engineer_messages)}")
        time.sleep(5)
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
            
            # Add explicit instruction for critic to use tools
            tool_instruction_message = TextMessage(
                content=f"""IMPORTANT: Before providing your assessment, you MUST:
1. Use the search_directory tool to find all output files: search_directory("{self._output_dir}", "*.png")
2. Use the analyze_plot tool on each visualization found: analyze_plot("{self._output_dir}/filename.png")
3. If no visualizations are found in the output directory, check what files exist: search_directory("{self._output_dir}", "*")
4. Examine the code and statistical evidence provided

Files are saved in the "{self._output_dir}" directory. Your review will NOT be considered complete unless you have used these tools to examine the visualizations and evidence.""",
                source="User"
            )
            messages_for_critic.append(tool_instruction_message)
            
            print(f"TOKEN ESTIMATE: critic team before run {revision_counter}: {estimate_tokens(messages_for_critic)}")
            print(f"NUM MESSAGES: {len(messages_for_critic)}")
            time.sleep(5)
            result_critic = await Console(self._critic_team.run_stream(task=messages_for_critic, cancellation_token=cancellation_token))
            critic_messages = result_critic.messages
            
            critic_messages = [message for message in critic_messages if isinstance(message, TextMessage)]
            print(f"TOKEN ESTIMATE: critic team after run {revision_counter}: {estimate_tokens(critic_messages)}")
            print(f"NUM MESSAGES: {len(critic_messages)}")
            time.sleep(5)
            
            # Check if the critic used any tools
            used_tools = False
            for msg in critic_messages:
                if ("search_directory" in msg.content and f"{self._output_dir}" in msg.content) and ("analyze_plot" in msg.content):
                    used_tools = True
                    break
            
            # If critic didn't use tools and this isn't already a retry for tools
            if not used_tools and not any("REMINDER: YOU MUST USE TOOLS" in msg.content for msg in messages_for_critic):
                print("Critic did not use the required tools. Sending reminder and retrying...")
                tool_reminder = TextMessage(
                    content=f"""REMINDER: YOU MUST USE TOOLS TO PROPERLY EVALUATE THE WORK.
1. First use search_directory("{self._output_dir}", "*.png") to find visualization files
2. Then use analyze_plot() on each visualization in the {self._output_dir} directory
3. Try search_directory("{self._output_dir}", "*") to see all files in the output directory
4. Only after examining the evidence should you provide your assessment

This is a requirement before approving or requesting revisions.""",
                    source="User"
                )
                messages_for_critic.append(tool_reminder)
                continue  # Skip to the next iteration of the loop to retry with the critic
            
            # Store the last message
            last_message_critic = critic_messages[-1]
            
            # Check for approval BEFORE removing tokens
            approves = self._critic_approve_token in last_message_critic.content
            revises = self._critic_revise_token in last_message_critic.content
            
            # Remove tokens after checking
            last_message_critic.content = last_message_critic.content.replace(self._critic_terminate_token, "")
            last_message_critic.content = last_message_critic.content.replace(self._critic_revise_token, "")
            last_message_critic.content = last_message_critic.content.replace(self._critic_approve_token, "")
            
            self.messages_to_summarize.append(last_message_critic)
            
            # Check if critic approves the work
            if approves:
                break
            elif not revises:
                print(f"Warning: Critic didn't provide a clear approval or revision token")
                # Continue anyway with revision
                
            revision_counter += 1
            if revision_counter > 3:
                break

            # Run the engineer team with the updated messages
            print(f"TOKEN ESTIMATE: engineer team before run {revision_counter}: {estimate_tokens(last_messages_engineer + [last_message_critic])}")
            print(f"NUM MESSAGES: {len(last_messages_engineer + [last_message_critic])}")
            time.sleep(5)
            
            # Add directory instruction before running engineer again
            directory_reminder = TextMessage(
                content=f"""IMPORTANT REMINDER: ALL output files (plots, data, etc.) MUST be saved in:
{self._output_dir}

Examples of correct paths:
- plt.savefig('{self._output_dir}/histogram.png')
- df.to_csv('{self._output_dir}/results.csv')""",
                source="User"
            )
            
            # Add troubleshooting reminder
            troubleshooting_reminder = TextMessage(
                content="""TROUBLESHOOTING REMINDER:

1. When fixing errors or addressing feedback:
   - Start by understanding exactly what's not working or what feedback needs to be addressed
   - Break down the problem into smaller parts
   - Test each part separately to find which component needs fixing
   - Make one change at a time and test its effect

2. For data splitting issues:
   - Check the distributions of key variables in each split
   - Make sure stratification is working correctly
   - Verify statistical similarity between splits with appropriate tests
   - Create clear tables showing the counts and percentages for key variables

3. For visualization issues:
   - Add proper titles, labels, and legends to all plots
   - Use appropriate color schemes
   - Include statistical context in the visualization
   - Save all plots to the correct output directory
""",
                source="User"
            )
            
            # Combine the messages with reminders
            engineer_iteration_messages = original_messages + last_messages_engineer + [last_message_critic, directory_reminder, troubleshooting_reminder]
            
            # Run the engineer team with updated messages
            result_engineer = await Console(self._engineer_team.run_stream(task=engineer_iteration_messages, cancellation_token=cancellation_token))
            engineer_messages = result_engineer.messages
            
            engineer_messages = [message for message in engineer_messages if isinstance(message, TextMessage)]
            engineer_messages = [message for message in engineer_messages if not "error" in message.content.lower()]
            print(f"TOKEN ESTIMATE: engineer team after run {revision_counter}: {estimate_tokens(engineer_messages)}")
            print(f"NUM MESSAGES: {len(engineer_messages)}")
            time.sleep(5)
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
            summary_message = TextMessage(content=summary_content, source="User")
            print(f"TOKEN ESTIMATE: summarizer agent: {estimate_tokens([summary_message])}")
            print(f"NUM MESSAGES: {len([summary_message])}")
            time.sleep(5)
            summary_result = await self._summarizer_agent.on_messages([summary_message], cancellation_token)
            final_result = summary_result.chat_message
            
            # Add a note about where the original implementation details can be found
            if isinstance(final_result, TextMessage):
                final_result.content += "\n\n(Note: This is a summary of the engineer's implementation and the critic's feedback. The full implementation details and code can be found in the previous messages.)"
        else:
            # If no summarizer agent, just return the last engineer message
            final_result = last_messages_engineer[-1] if last_messages_engineer else None
            
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