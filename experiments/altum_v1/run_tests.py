#!/usr/bin/env python3
import os
import sys
import pytest

if __name__ == "__main__":
    # Add the parent directory to the Python path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(parent_dir))
    
    # Run pytest on the tests directory
    test_dir = os.path.join(parent_dir, "tests")
    print(f"Running tests in: {test_dir}")
    
    # Run the tests
    result = pytest.main(["-v", test_dir])
    
    sys.exit(result) 