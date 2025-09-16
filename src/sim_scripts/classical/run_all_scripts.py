# import subprocess

# # List of Python scripts to run
# scripts = [
#     '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/baart_ito.py',
#     '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/blur_ito.py',
#     '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/deriv2_ito.py',
#     "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/foxgood_ito.py",
#     "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/phillips_ito.py",
#     "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/wing_ito.py",
#     "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/shaw_ito.py",
#     '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/heat_ito.py',
#     '/Users/steveh/Downloads/NIH 23-24/LocReg_Python/classical_problems/TestingIto/gravity_ito.py'
# ]

# # Run each script
# for script in scripts:
#     try:
#         result = subprocess.run(['python', script], capture_output=True, text=True, check=True)
#         print(f"Output of {script}:\n{result.stdout}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running {script}:\n{e.stderr}")
# import os
# import concurrent.futures
# from tqdm import tqdm

# cwd = os.getcwd()

# # List of Python scripts to run
# scripts = [
#     f'{cwd}/baart_ito.py',
#     f'{cwd}/blur_ito.py',
#     f'{cwd}/deriv2_ito.py',
#     f'{cwd}/foxgood_ito.py',
#     f'{cwd}/phillips_ito.py',
#     f'{cwd}/wing_ito.py',
#     f'{cwd}/shaw_ito.py',
#     f'{cwd}/heat_ito.py',
#     f'{cwd}/gravity_ito.py'
# ]

# def run_script(script):
#     print(f"Starting {script}")
#     try:
#         result = os.system(f'python3 {script}')
#         if result != 0:
#             print(f"Error running {script}")
#     except Exception as e:
#         print(f"Exception occurred while running {script}: {e}")
#     print(f"Completed {script}")

# if __name__ == '__main__':
#     # Adjust the number of workers to be within the system limit
#     max_workers = min(os.cpu_count(), len(scripts))
    
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {executor.submit(run_script, script): script for script in scripts}
        
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(scripts)):
#             script = futures[future]
#             try:
#                 future.result()
#             except Exception as exc:
#                 print(f'{script} generated an exception: {exc}')
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cwd = os.getcwd()

# List of Python scripts to run
scripts = [
    # f'{cwd}/baart_ito.py',
    f'{cwd}/blur_ito.py',
    f'{cwd}/deriv2_ito.py',
    f'{cwd}/foxgood_ito.py',
    f'{cwd}/phillips_ito.py',
    f'{cwd}/wing_ito.py',
    f'{cwd}/shaw_ito.py',
    f'{cwd}/heat_ito.py',
    f'{cwd}/gravity_ito.py'
]

# def run_script(script):
#     print(f"Starting {script}")
#     try:
#         result = os.system(f'python3 {script}')
#         if result != 0:
#             print(f"Error running {script}")
#     except Exception as e:
#         print(f"Exception occurred while running {script}: {e}")
#     print(f"Completed {script}")

# if __name__ == '__main__':
#     # Determine number of CPU cores
#     num_cores = os.cpu_count() or 1  # If unable to determine, default to 1 core

#     # Use ThreadPoolExecutor to run scripts concurrently
#     with ThreadPoolExecutor(max_workers=num_cores) as executor:
#         futures = [executor.submit(run_script, script) for script in scripts]

#         # Display progress with tqdm
#         for future in tqdm(as_completed(futures), total=len(scripts), desc="Running scripts concurrently"):
#             pass  # Wait for completion


def run_script(script):
    print(f"Starting {script}")
    try:
        result = os.system(f'python3 {script}')
        if result != 0:
            print(f"Error running {script}")
    except Exception as e:
        print(f"Exception occurred while running {script}: {e}")
    print(f"Completed {script}")

if __name__ == '__main__':
    for script in tqdm(scripts, desc="Running scripts sequentially"):
        run_script(script)
# import concurrent.futures
# import time

# def process_task(i):
#     # Simulate a task that takes time
#     time.sleep(0.1)
#     return f"Task {i} completed"

# def main():
#     tasks = range(50)  # 50 iterations
#     results = []

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = {executor.submit(process_task, i): i for i in tasks}
#         for future in concurrent.futures.as_completed(futures):
#             results.append(future.result())

#     for result in results:
#         print(result)

# if __name__ == "__main__":
#     main()