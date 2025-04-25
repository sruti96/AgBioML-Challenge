#!/bin/bash
# Run script for Altum v2 pipeline

# Ensure we're in the project root
cd "$(dirname "$0")/../.." || { echo "Error: Could not change to project root"; exit 1; }

# Create necessary directories if they don't exist
mkdir -p memory_v2 task_outputs

# Set up Python environment
# Uncomment and modify if needed
# source path/to/your/venv/bin/activate

# Run the main pipeline script
python experiments/altum_v2/01_run_pipeline_v2.py "$@"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Altum v2 pipeline completed successfully!"
    echo "Results are available in the task_outputs directory."
    echo "The lab notebook is available at memory_v2/lab_notebook.txt."
else
    echo "Error: Altum v2 pipeline exited with an error."
    echo "Check the logs for details."
fi 