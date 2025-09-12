import numpy as np
from concurrent.futures import ProcessPoolExecutor

def generate_random_numbers(seed, size):
    rng = np.random.default_rng(seed)
    return rng.random(size)

def main():
    M = 10  # Number of parallel tasks
    size = 1000  # Size of each random number vector

    # Create a SeedSequence
    seed_seq = np.random.SeedSequence(12345)
    # Spawn M child seeds
    child_seeds = seed_seq.spawn(M)

    with ProcessPoolExecutor(max_workers=M) as executor:
        futures = [executor.submit(generate_random_numbers, seed, size) for seed in child_seeds]
        results = [future.result() for future in futures]

    # 'results' now contains M unique random number vectors
    for i, result in enumerate(results):
        print(f"Random numbers from task {i}: {result[:5]}...")  # Displaying first 5 numbers for brevity

if __name__ == "__main__":
    main()
