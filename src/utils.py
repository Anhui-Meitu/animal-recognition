import pathlib
import os

def get_project_root():
    """
    Get the root directory of the project.
    """
    # Get the current file's directory
    current_file = pathlib.Path(__file__).resolve()
    # Get the root directory by going up two levels
    project_root = current_file.parents[2]
    return project_root