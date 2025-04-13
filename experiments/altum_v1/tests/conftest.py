import os
import sys

# Add the parent directory to the Python path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# This helps pytest discover the modules properly
pytest_plugins = [] 