import os
import json
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock

# Update import path to correctly reference the utils module
from utils import (
    save_messages_structured,
    get_structured_summaries,
    format_structured_task_prompt,
    get_workflow_state,
    update_workflow_state,
    save_workflow_checkpoint
)

pytestmark = pytest.mark.asyncio  # Mark all tests in this module as asyncio tests

class TestWorkflowIntegration:
    def setup_directories(self):
        """Ensure all required directories exist for testing."""
        # Create output directories that might be needed
        os.makedirs(os.path.join(self.temp_dir, 'task_3_workdir'), exist_ok=True)
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a temporary directory for test files and clean up after tests."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Make sure the directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create required subdirectories
        self.setup_directories()
        
        # Mock the memory directory with the full module path
        with patch('utils.memory_dir', self.temp_dir):
            yield
            
        # Clean up after test
        shutil.rmtree(self.temp_dir)
    
    @pytest.fixture
    def mock_messages(self):
        """Create mock messages for testing."""
        def _create_messages(content):
            message = MagicMock()
            message.dump.return_value = {"content": content}
            return [message]
        return _create_messages
    
    async def create_full_workflow_state(self, mock_messages):
        """Set up a complete workflow state to simulate the full process."""
        # Stage 2 (EDA), all subtasks with 1 iteration each
        for subtask in range(1, 4):
            await save_messages_structured(
                2, subtask, 1, 
                mock_messages(f"EDA Stage, Subtask {subtask} message"),
                f"EDA Stage, Subtask {subtask} summary with plot_eda_{subtask}.png",
                f"EDA Stage, Subtask {subtask} task description"
            )
            update_workflow_state(2, subtask, 1)
        
        # Mark EDA stage as completed
        state = get_workflow_state()
        if "stages_completed" not in state:
            state["stages_completed"] = []
        if 2 not in state["stages_completed"]:
            state["stages_completed"].append(2)
        
        with open(os.path.join(self.temp_dir, 'workflow_state.json'), 'w') as f:
            json.dump(state, f)
        
        # Stage 3 (DATA_SPLIT), subtask 1 with 1 iteration
        await save_messages_structured(
            3, 1, 1,
            mock_messages("DATA_SPLIT Stage, Subtask 1 message"),
            "DATA_SPLIT Stage, Subtask 1 summary. Team A decided to split by dataset.",
            "DATA_SPLIT Stage, Subtask 1 task description"
        )
        update_workflow_state(3, 1, 1)
        
        # Stage 3 (DATA_SPLIT), subtask 2 with 2 iterations - this is where the bug happens
        await save_messages_structured(
            3, 2, 1,
            mock_messages("DATA_SPLIT Stage, Subtask 2, iteration 1 message"),
            "DATA_SPLIT Stage, Subtask 2, iteration 1 summary. Created plots: dataset_split_sizes.png, age_distribution_datasets.png",
            "DATA_SPLIT Stage, Subtask 2, iteration 1 task description"
        )
        update_workflow_state(3, 2, 1)
        
        # Second iteration with DIFFERENT plot names
        await save_messages_structured(
            3, 2, 2,
            mock_messages("DATA_SPLIT Stage, Subtask 2, iteration 2 message"),
            "DATA_SPLIT Stage, Subtask 2, iteration 2 summary. Created revised plots: train_split_distribution_rev.png, val_split_distribution_rev.png, test_split_distribution_rev.png",
            "DATA_SPLIT Stage, Subtask 2, iteration 2 task description"
        )
        update_workflow_state(3, 2, 2)
        
        # Save a checkpoint to simulate workflow advancement
        save_workflow_checkpoint(3, 2, 2, "Completed iteration 2 of subtask 2")
        
    async def test_full_workflow_iteration_integrity(self, mock_messages):
        """Test a complete workflow to ensure iteration integrity across stages."""
        # Set up the complete workflow state
        await self.create_full_workflow_state(mock_messages)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Now prepare for subtask 3 (review by Team A)
            # This should get the latest iteration of subtask 2 (iteration 2)
            update_workflow_state(3, 3, 1)  # Moving to subtask 3
            
            # Format the prompt for Team A in subtask 3
            task_text = "Review the implementation from the engineer."
            formatted_prompt = await format_structured_task_prompt(3, 3, task_text, 1)
            
            # Verify the formatted prompt contains the CORRECT (latest iteration) plot names
            assert "train_split_distribution_rev.png" in formatted_prompt
            assert "val_split_distribution_rev.png" in formatted_prompt
            assert "test_split_distribution_rev.png" in formatted_prompt
            
            # Verify it does NOT contain the older plot names
            assert "dataset_split_sizes.png" not in formatted_prompt
            assert "age_distribution_datasets.png" not in formatted_prompt
    
    async def test_get_maximum_iteration_integrity(self):
        """Test the get_maximum_iteration function to ensure it correctly identifies the latest iteration."""
        from utils import get_maximum_iteration
        
        with patch('utils.memory_dir', self.temp_dir):
            # Create a test summary file with multiple iterations
            summary_file = os.path.join(self.temp_dir, 'structured_summaries.json')
            os.makedirs(self.temp_dir, exist_ok=True)
            
            test_data = {
                "stage3": {
                    "subtask2": {
                        "iteration1": {"summary": "First iteration"},
                        "iteration2": {"summary": "Second iteration"},
                        "iteration3": {"summary": "Third iteration"}
                    }
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(test_data, f)
            
            # Get the maximum iteration
            max_iter = get_maximum_iteration(3, 2)
            
            # Verify it returns the correct maximum
            assert max_iter == 3
    
    async def test_bug_simulation_in_main_function(self, mock_messages):
        """Simulates the actual workflow in the main function to reproduce the bug."""
        # Mock the runsubtask1, runsubtask2, runsubtask3 functions
        # First, create the essential workflow state
        await self.create_full_workflow_state(mock_messages)
        
        # Now simulate what happens in the main function
        with patch('utils.memory_dir', self.temp_dir):
            # In the main function, after subtask 2 completes:
            iteration = 2  # We completed 2 iterations in subtask 2
            
            # Save checkpoint after subtask 2
            save_workflow_checkpoint(3, 2, iteration, f"Data Splitting Implementation (Iteration {iteration})")
            
            # Move to subtask 3
            update_workflow_state(3, 3, 1)
            
            # Simulate formatting prompt for subtask 3, which should include latest iteration of subtask 2
            prompt_text = "Review the implementation from the engineer"
            formatted = await format_structured_task_prompt(3, 3, prompt_text, 1)
            
            # Verify it has the right plot names (from iteration 2, not iteration 1)
            assert "train_split_distribution_rev.png" in formatted
            assert "dataset_split_sizes.png" not in formatted
    
    async def test_fixed_get_structured_summaries(self):
        """Test a fixed version of get_structured_summaries to ensure it works correctly."""
        # This test implements a corrected version of get_structured_summaries
        
        # First create our test data
        summary_file = os.path.join(self.temp_dir, 'structured_summaries.json')
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        test_data = {
            "stage3": {
                "subtask1": {
                    "iteration1": {
                        "summary": "Subtask 1 summary"
                    }
                },
                "subtask2": {
                    "iteration1": {
                        "summary": "Subtask 2, iteration 1 with old_plot.png"
                    },
                    "iteration2": {
                        "summary": "Subtask 2, iteration 2 with new_plot.png"
                    }
                }
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(test_data, f)
        
        async def fixed_get_structured_summaries(current_stage, current_subtask, current_iteration):
            """Corrected implementation to always get latest iteration."""
            try:
                with open(summary_file, 'r') as f:
                    all_summaries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return "No previous summaries available."
            
            result_parts = []
            
            # Add all completed subtasks from current stage
            stage_key = f"stage{current_stage}"
            if stage_key in all_summaries:
                for subtask_key, subtask_data in sorted(all_summaries[stage_key].items()):
                    subtask_num = int(subtask_key.replace("subtask", ""))
                    
                    # Include prior subtasks
                    if subtask_num < current_subtask:
                        # Get the latest iteration for this subtask
                        iterations = [int(k.replace("iteration", "")) for k in subtask_data.keys()]
                        max_iteration = max(iterations)
                        iter_key = f"iteration{max_iteration}"
                        
                        summary_data = subtask_data[iter_key]
                        result_parts.append(f"Latest from {subtask_key}: {summary_data['summary']}")
            
            return "\n".join(result_parts)
        
        # Test the fixed implementation
        with patch('utils.memory_dir', self.temp_dir):
            # Get summaries for subtask 3 (which should get latest from subtask 2)
            result = await fixed_get_structured_summaries(3, 3, 1)
            
            # Verify it has the latest iteration's plot
            assert "new_plot.png" in result
            assert "old_plot.png" not in result


# Proposed fix for the bug
async def fixed_get_structured_summaries(current_stage, current_subtask, current_iteration):
    """Fixed implementation of get_structured_summaries that always gets the latest iteration.
    
    This function ensures that when retrieving summaries from previous subtasks, it always
    uses the latest iteration of each subtask, not just any arbitrary iteration.
    """
    memory_dir = 'memory'
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