import numpy as np
from src.regularization.reg_methods import nnls, gcv
from src.utils import simulations_functions as sf

def test_pipeline_synthetic_nnls():
    # Step 1: generate small synthetic dataset
    signal = sf.generate_signal(n_points=50, noise_level=0.01)
    
    # Step 2: run NNLS on it
    distribution = nnls.solve(signal, n_components=20)
    
    # Step 3: sanity checks
    assert distribution.shape == (20,)
    assert np.all(distribution >= 0)  # non-negativity
    assert np.isclose(distribution.sum(), 1, rtol=0.2)  # normalized-ish

def test_pipeline_gcv_regularization():
    signal = sf.generate_signal(n_points=30, noise_level=0.05)
    best_lambda, solution = gcv.select_lambda(signal)
    
    assert isinstance(best_lambda, float)
    assert solution.shape[0] > 0