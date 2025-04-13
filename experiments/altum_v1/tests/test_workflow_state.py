import os
import json
import pytest
import shutil
import tempfile
from unittest.mock import patch, MagicMock

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

class TestWorkflowState:
    def setup_directories(self):
        """Ensure all required directories exist for testing."""
        # Create output directories that might be needed
        os.makedirs(os.path.join(self.temp_dir, 'task_3_workdir'), exist_ok=True)
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Create a temporary directory for test files and clean up after tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_memory_dir = 'memory'
        
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
        message1 = MagicMock()
        message1.dump.return_value = {"content": "Message 1 content"}
        
        message2 = MagicMock()
        message2.dump.return_value = {"content": "Message 2 content"}
        
        return [message1, message2]
    
    @pytest.fixture
    def create_test_summaries(self):
        """Helper to create test summary data in the temp directory."""
        def _create_summaries(iterations=1):
            summary_file = os.path.join(self.temp_dir, 'structured_summaries.json')
            test_data = {}
            
            # Create test data for multiple stages and subtasks
            for stage in [2, 3]:  # EDA and DATA_SPLIT stages
                stage_key = f"stage{stage}"
                test_data[stage_key] = {}
                
                for subtask in [1, 2, 3]:
                    subtask_key = f"subtask{subtask}"
                    test_data[stage_key][subtask_key] = {}
                    
                    for iteration in range(1, iterations + 1):
                        iter_key = f"iteration{iteration}"
                        test_data[stage_key][subtask_key][iter_key] = {
                            "timestamp": f"2023-01-0{iteration}T12:00:00",
                            "iteration": iteration,
                            "task_description": f"Task for {stage_key} {subtask_key} iteration {iteration}",
                            "summary": (f"Summary for {stage_key} {subtask_key} iteration {iteration}. "
                                       f"Plot: plot_{stage}_{subtask}_{iteration}.png")
                        }
            
            # Save to file
            os.makedirs(self.temp_dir, exist_ok=True)
            with open(summary_file, 'w') as f:
                json.dump(test_data, f)
                
            return test_data
        
        return _create_summaries
    
    @pytest.fixture
    def create_workflow_state(self):
        """Helper to create test workflow state data."""
        def _create_state(current_stage=3, subtask=2, iteration=2):
            state_file = os.path.join(self.temp_dir, 'workflow_state.json')
            state = {
                "current_stage": current_stage,
                "stages_completed": [1, 2] if current_stage > 2 else [1],
                "iterations": {
                    f"stage{current_stage}": {
                        f"subtask{subtask}": iteration
                    }
                }
            }
            
            # Save to file
            os.makedirs(self.temp_dir, exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(state, f)
                
            return state
        
        return _create_state
    
    @pytest.mark.asyncio
    async def test_save_messages_structured(self, mock_messages):
        """Test that save_messages_structured correctly saves to the right structure."""
        stage = 3
        subtask = 2
        iteration = 2
        summary = "Summary of the work done in this iteration"
        task_description = "Task description for data splitting"
        
        with patch('utils.memory_dir', self.temp_dir):
            await save_messages_structured(stage, subtask, iteration, mock_messages, summary, task_description)
            
            # Verify saved messages
            messages_file = os.path.join(self.temp_dir, 'structured_messages.json')
            with open(messages_file, 'r') as f:
                saved_messages = json.load(f)
            
            assert f"stage{stage}" in saved_messages
            assert f"subtask{subtask}" in saved_messages[f"stage{stage}"]
            assert f"iteration{iteration}" in saved_messages[f"stage{stage}"][f"subtask{subtask}"]
            assert len(saved_messages[f"stage{stage}"][f"subtask{subtask}"][f"iteration{iteration}"]) == 2
            
            # Verify saved summary
            summary_file = os.path.join(self.temp_dir, 'structured_summaries.json')
            with open(summary_file, 'r') as f:
                saved_summaries = json.load(f)
                
            assert f"stage{stage}" in saved_summaries
            assert f"subtask{subtask}" in saved_summaries[f"stage{stage}"]
            assert f"iteration{iteration}" in saved_summaries[f"stage{stage}"][f"subtask{subtask}"]
            assert saved_summaries[f"stage{stage}"][f"subtask{subtask}"][f"iteration{iteration}"]["summary"] == summary
            assert saved_summaries[f"stage{stage}"][f"subtask{subtask}"][f"iteration{iteration}"]["task_description"] == task_description
    
    @pytest.mark.asyncio
    async def test_save_multiple_iterations(self, mock_messages):
        """Test saving multiple iterations and verify all are preserved."""
        stage = 3
        subtask = 2
        
        with patch('utils.memory_dir', self.temp_dir):
            # Save iteration 1
            await save_messages_structured(stage, subtask, 1, mock_messages, 
                                          "Summary for iteration 1 with plot_3_2_1.png", 
                                          "Task for iteration 1")
            
            # Save iteration 2 - with different plot name
            await save_messages_structured(stage, subtask, 2, mock_messages, 
                                          "Summary for iteration 2 with revised_plot_3_2_2.png", 
                                          "Task for iteration 2")
            
            # Verify both iterations saved
            summary_file = os.path.join(self.temp_dir, 'structured_summaries.json')
            with open(summary_file, 'r') as f:
                saved_summaries = json.load(f)
            
            assert "iteration1" in saved_summaries[f"stage{stage}"][f"subtask{subtask}"]
            assert "iteration2" in saved_summaries[f"stage{stage}"][f"subtask{subtask}"]
            assert "plot_3_2_1.png" in saved_summaries[f"stage{stage}"][f"subtask{subtask}"]["iteration1"]["summary"]
            assert "revised_plot_3_2_2.png" in saved_summaries[f"stage{stage}"][f"subtask{subtask}"]["iteration2"]["summary"]
    
    @pytest.mark.asyncio
    async def test_get_structured_summaries_latest_iteration(self, create_test_summaries):
        """Test that get_structured_summaries retrieves the latest iteration."""
        # Create test data with 3 iterations
        create_test_summaries(iterations=3)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Get summaries for current stage=3, subtask=2
            summaries = await get_structured_summaries(current_stage=3, current_subtask=2, current_iteration=1)
            
            # Verify it includes the latest iteration from previous subtasks
            assert "plot_3_1_3.png" in summaries  # Should get latest iteration (3) of subtask 1
            assert "plot_2_3_3.png" in summaries  # Should get latest iteration (3) of previous stage
    
    @pytest.mark.asyncio
    async def test_get_structured_summaries_include_prev_iterations(self, create_test_summaries):
        """Test that get_structured_summaries includes previous iterations of the current subtask."""
        # Create test data with 3 iterations
        create_test_summaries(iterations=3)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Get summaries for current stage=3, subtask=2, iteration=3
            summaries = await get_structured_summaries(current_stage=3, current_subtask=2, current_iteration=3)
            
            # Should include previous iterations of current subtask
            assert "plot_3_2_1.png" in summaries  # Iteration 1 of current subtask
            assert "plot_3_2_2.png" in summaries  # Iteration 2 of current subtask
    
    @pytest.mark.asyncio
    async def test_format_structured_task_prompt(self, create_test_summaries):
        """Test that format_structured_task_prompt correctly includes relevant summaries."""
        # Create test data
        create_test_summaries(iterations=2)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Format prompt for stage=3, subtask=3 (which should get results from stage=3, subtask=2)
            task_text = "New task for subtask 3"
            formatted = await format_structured_task_prompt(3, 3, task_text, 1)
            
            # Verify it includes the summary from the previous subtask
            assert "Summary for stage3 subtask2 iteration 2" in formatted
            assert "plot_3_2_2.png" in formatted  # Latest iteration's plot from previous subtask
            assert task_text in formatted
    
    @pytest.mark.asyncio
    async def test_format_task_prompt_with_previous_iteration(self, create_test_summaries):
        """Test that format_structured_task_prompt includes previous iteration when on iteration > 1."""
        # Create test data
        create_test_summaries(iterations=3)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Format prompt for iteration 2 of stage=3, subtask=2
            task_text = "Iteration 2 task text"
            formatted = await format_structured_task_prompt(3, 2, task_text, 2)
            
            # Verify it includes the previous iteration's summary
            assert "Summary for stage3 subtask2 iteration 1" in formatted
            
    @pytest.mark.asyncio
    async def test_latest_iteration_takes_precedence(self, create_test_summaries, create_workflow_state):
        """Test comprehensive scenario: when workflow moves from subtask 2 to 3, verify latest iteration of subtask 2 is used."""
        # Create test data with 3 iterations
        create_test_summaries(iterations=3)
        
        # Create workflow state showing we're now on subtask 3 after completing subtask 2
        create_workflow_state(current_stage=3, subtask=3, iteration=1)
        
        with patch('utils.memory_dir', self.temp_dir):
            # Format prompt for subtask 3
            task_text = "Review the implementation from the engineer"
            formatted = await format_structured_task_prompt(3, 3, task_text, 1)
            
            # Very specifically check this has the plot from the LATEST iteration of subtask 2
            assert "plot_3_2_3.png" in formatted
            assert "Summary for stage3 subtask2 iteration 3" in formatted
            
            # Should not contain plots from earlier iterations
            assert "plot_3_2_1.png" not in formatted
            assert "plot_3_2_2.png" not in formatted
    
    @pytest.mark.asyncio
    async def test_bug_reproduction(self, mock_messages):
        """Test specifically reproducing the bug where wrong plot names appear in summaries."""
        stage = 3  # DATA_SPLIT_STAGE
        subtask_2 = 2
        subtask_3 = 3
        
        with patch('utils.memory_dir', self.temp_dir):
            # Iteration 1: Engineer creates plots with certain names
            engineer_summary_1 = ("Implementation complete. Created plots: "
                                "dataset_split_sizes.png, age_distribution_datasets.png")
            
            await save_messages_structured(stage, subtask_2, 1, mock_messages, 
                                         engineer_summary_1, "Task for engineer iteration 1")
            
            # Iteration 2: Engineer creates plots with different names
            engineer_summary_2 = ("Revised implementation. Created plots: "
                                 "train_split_distribution_rev.png, val_split_distribution_rev.png, "
                                 "test_split_distribution_rev.png")
            
            await save_messages_structured(stage, subtask_2, 2, mock_messages, 
                                         engineer_summary_2, "Task for engineer iteration 2")
            
            # Update workflow state to iteration 2
            update_workflow_state(stage, subtask_2, 2)
            
            # Now the workflow moves to subtask 3, iteration 1
            summary_for_team_a = await get_structured_summaries(stage, subtask_3, 1)
            
            # BUG: The summary should have the LATEST iteration's plot names
            assert "train_split_distribution_rev.png" in summary_for_team_a
            assert "dataset_split_sizes.png" not in summary_for_team_a  # Should not have iteration 1 plot names
    
    @pytest.mark.asyncio
    async def test_saving_checkout_preserves_iterations(self):
        """Test that saving and loading checkpoints preserves all iteration information."""
        with patch('utils.memory_dir', self.temp_dir):
            # Set up initial state with multiple iterations
            update_workflow_state(3, 2, 1)  # Stage 3, subtask 2, iteration 1
            
            # Save a checkpoint
            checkpoint_id = save_workflow_checkpoint(3, 2, 1, "First iteration")
            
            # Update to iteration 2
            update_workflow_state(3, 2, 2)
            
            # Save another checkpoint 
            checkpoint_id2 = save_workflow_checkpoint(3, 2, 2, "Second iteration")
            
            # Verify checkpoints saved the correct iteration
            checkpoint_file = os.path.join(self.temp_dir, 'workflow_checkpoints.json')
            with open(checkpoint_file, 'r') as f:
                checkpoints = json.load(f)
            
            assert checkpoints["checkpoints"][checkpoint_id]["iteration"] == 1
            assert checkpoints["checkpoints"][checkpoint_id2]["iteration"] == 2
            
            # Verify the workflow state has the correct latest iteration
            state = get_workflow_state()
            assert state["iterations"]["stage3"]["subtask2"] == 2 