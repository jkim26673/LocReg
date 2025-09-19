# Helper functions for run-specific paths
import os
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

def run_dir(root_dir:str, *args: str):
    """
    Returns a results subdirectory for a specific experiment/run.
    Creates the directory if it doesn't exist.
    """
    path = gen_subdirpath(root_dir, *args)
    create_dir(path)
    return path

def get_filepath(file_name:str):
    try:
        os.path.exists(file_name)
    except:
        raise AssertionError("File not found.")
    return os.path.abspath(file_name)