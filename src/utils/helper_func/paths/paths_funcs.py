# Helper functions for run-specific paths
import os
from glob import glob
def gen_subdirpath(root_dir: str, *args:str):
    """
    Returns path for subdirectory for given main directory
    :params: root_dir: root directory
    :params: *args: additional arguments of subdirectories
    :returns: str : pathname

    ex. gen_subdir(results_dir, "brain)
    """
    path = os.path.join(root_dir, *args)
    return path

def create_dir(path: str):
    """
    Checks and creates the directory if it doesn't exist.
    :params: path: file path name
    :returns: str: path
    """
    os.makedirs(path, exist_ok=True)
    return 

def gen_results_dir(root_dir:str, *args: str):
    """
    Returns a results subdirectory for a specific experiment/run.
    Creates the directory if it doesn't exist.
    """
    path = gen_subdirpath(root_dir, *args)
    create_dir(path)
    return path

def get_filepath(file_name:str):
    matches = glob(f"**/{file_name}", recursive=True)
    if not matches:
        raise AssertionError(f"File not found {file_name}.")
    return os.path.abspath(matches[0])

