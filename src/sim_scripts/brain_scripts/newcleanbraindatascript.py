#!/usr/bin/env python3
"""
Optimized Brain Data Processing Script
Performs T2 relaxometry analysis using multiple regularization methods.
"""
import os
import sys
sys.path.append(".")
import logging
import pickle
import time
from datetime import date
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.linalg import norm as linalg_norm
from scipy.optimize import nnls
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Simulations.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
# from Simulations.upenzama import UPEN_Zama
from src.utils.load_imports.loading import *

# import matlab.engine
# import mosek
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# logging.info(f'MOSEK License Set from {mosek_license_path}')

@dataclass
class Config:
    """Configuration parameters for the analysis."""
    # Data paths
    brain_data_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat"
    snr_map_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat"
    mask_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"
    log_dir: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1"
    matlab_path: str = r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test'
    
    # Analysis parameters
    dte: float = 11.3
    n_te: int = 32
    n_t2: int = 150
    t2_min: float = 10.0
    t2_max: float = 200.0
    myelin_threshold: float = 40.0
    data_threshold: float = 50.0
    
    # Regularization parameters
    lambda_min: float = 1e-5
    lambda_max: float = 2
    n_lambda: int = 50
    gamma_init: float = 0.5
    max_iter: int = 50
    
    # Noise parameters
    uniform_noise: bool = True
    num_noise_signals: int = 1000
    noise_coords_x_range: Tuple[int, int] = (130, 200)
    noise_coords_y_range: Tuple[int, int] = (100, 200)
    tail_length: int = 3
    
    # Processing parameters
    checkpoint_interval: int = 1000
    checkpoint_time_interval: int = 900
    
    # Parallel processing parameters
    n_processes: Optional[int] = None  # None means use all available cores
    chunk_size: int = 50  # Number of voxels per chunk
    
    # MWF threshold for considering as zero
    mwf_zero_threshold: float = 1e-6


def process_voxel_chunk(chunk_data):
    """Process a chunk of voxels. This function runs in parallel processes."""
    (voxel_coords, brain_data_chunk, snr_map_chunk, config_dict, 
     A, te, t2, dt, lambda_range, myelin_idx, uniform_noise_val) = chunk_data
    
    # Reconstruct config object
    config = Config(**config_dict)
    
    results = []
    
    for x, y in voxel_coords:
        try:
            result = process_single_voxel(
                x, y, brain_data_chunk, snr_map_chunk, config,
                A, te, t2, dt, lambda_range, myelin_idx, uniform_noise_val
            )
            if result is not None:
                results.append(result)
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing voxel ({x}, {y}): {e}")
            continue
    
    return results


def process_single_voxel(x, y, brain_data, snr_map, config, A, te, t2, dt, 
                        lambda_range, myelin_idx, uniform_noise_val):
    """Process a single voxel and return results."""
    # Check data threshold
    if brain_data[x, y, 0] < config.data_threshold:
        return None
        
    try:
        curr_data = brain_data[x, y, :].copy()
        curr_snr = snr_map[x, y]
        
        # Normalize data
        curr_data = normalize_data(curr_data, A, dt)
        
        # Get reference solution (GCV without noise)
        ref_rec, ref_lamb = GCV_NNLS(curr_data, A, lambda_range)
        ref_rec = ref_rec[:, 0]
        ref_lamb = np.squeeze(ref_lamb)
        ref_rec, ref_mwf, ref_flag = compute_mwf_filtered(ref_rec, dt, myelin_idx)
        
        # Add noise
        data_gt = A @ ref_rec
        noisy_data = data_gt + uniform_noise_val
        
        # Apply regularization methods
        results = apply_regularization_methods(noisy_data, ref_lamb, A, config, dt, myelin_idx)
        
        # Create result dataframe
        result_df = create_result_dataframe(
            x, y, noisy_data, curr_snr, ref_rec, ref_mwf, ref_lamb, ref_flag, 
            results, uniform_noise_val, config.n_t2
        )
        
        return result_df
        
    except Exception as e:
        print(f"Error processing voxel ({x}, {y}): {e}")
        return None


def normalize_data(data: np.ndarray, A: np.ndarray, dt: float) -> np.ndarray:
    """Normalize data using NNLS solution."""
    sol = nnls(A, data)[0]
    factor = np.sum(sol) * dt
    return data / factor if factor > 0 else data


def apply_regularization_methods(data: np.ndarray, ref_lamb: float, A: np.ndarray, 
                               config: Config, dt: float, myelin_idx) -> Dict[str, Any]:
    """Apply all regularization methods to the data."""
    results = {}
    lambda_range = np.logspace(
        np.log10(config.lambda_min), 
        np.log10(config.lambda_max), 
        config.n_lambda
    ).reshape(-1, 1)
    
    # GCV
    gcv_rec, gcv_lamb = GCV_NNLS(data, A, lambda_range)
    gcv_rec = gcv_rec[:, 0]
    gcv_lamb = np.squeeze(gcv_lamb)
    results['GCV'] = {
        'estimate': gcv_rec,
        'lambda': gcv_lamb,
        'mwf': None,
        'flag': None
    }
    
    # LocReg variants
    ini_lamb = gcv_lamb
    
    # Standard LocReg
    lr_rec, lr_lamb, _, _, _ = LocReg_Ito_mod(
        data, A, ini_lamb, config.gamma_init, maxiter=config.max_iter
    )
    results['LR'] = {
        'estimate': lr_rec,
        'lambda': lr_lamb,
        'mwf': None,
        'flag': None
    }
    
    # LocReg with 1st derivative
    lr1d_rec, lr1d_lamb, _, _, _ = LocReg_Ito_mod_deriv(
        data, A, ini_lamb, config.gamma_init, maxiter=config.max_iter
    )
    results['LR1D'] = {
        'estimate': lr1d_rec,
        'lambda': lr1d_lamb,
        'mwf': None,
        'flag': None
    }
    
    # LocReg with 2nd derivative
    lr2d_rec, lr2d_lamb, _, _, _ = LocReg_Ito_mod_deriv2(
        data, A, ini_lamb, config.gamma_init, maxiter=config.max_iter
    )
    results['LR2D'] = {
        'estimate': lr2d_rec,
        'lambda': lr2d_lamb,
        'mwf': None,
        'flag': None
    }
    
    # UPEN
    std_dev = np.std(data[-5:])
    snr_est = np.max(np.abs(data)) / std_dev
    threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(data) / snr_est
    
    upen_rec, upen_lamb = UPEN_Zama(
        A, data, gcv_rec, threshold, 1e-3, 50, 1e-5
    )
    results['UPEN'] = {
        'estimate': upen_rec,
        'lambda': upen_lamb,
        'mwf': None,
        'flag': None
    }
    
    # Compute MWF and flags for all methods
    for method in results:
        rec, mwf, flag = compute_mwf_filtered(results[method]['estimate'], dt, myelin_idx)
        results[method]['estimate'] = rec
        results[method]['mwf'] = mwf
        results[method]['flag'] = flag

    return results


def compute_mwf_filtered(reconstruction: np.ndarray, dt: float, myelin_idx, 
                        tol: float = 1e-6) -> Tuple[np.ndarray, float, int]:
    """Compute MWF with filtering for edge cases."""
    if np.all(np.abs(reconstruction) < tol) or np.all(reconstruction[:-1] == 0):
        return reconstruction, 0.0, 1
        
    try:
        total_sum = np.sum(reconstruction) * dt
        if total_sum == 0:
            return reconstruction, 0.0, 1
            
        f_normalized = reconstruction / total_sum
        total_mwf = np.cumsum(f_normalized)
        mwf = total_mwf[myelin_idx[0][-1]]
        return f_normalized, mwf, 0
        
    except (ZeroDivisionError, IndexError):
        return reconstruction, 0.0, 1


def create_result_dataframe(x: int, y: int, data: np.ndarray, snr: float,
                           ref_rec: np.ndarray, ref_mwf: float, ref_lamb: float,
                           ref_flag: int, results: Dict[str, Any], uniform_noise_val: np.ndarray,
                           n_t2: int) -> pd.DataFrame:
    """Create result dataframe for a single voxel."""
    row_data = {
        "X_val": x,
        "Y_val": y,
        "curr_data": data,
        "noise": uniform_noise_val,
        "curr_SNR": snr,
        "ref_estimate": ref_rec,
        "MWF_Ref": ref_mwf,
        "Lam_Ref": ref_lamb,
        "Ref_Flag_Val": ref_flag
    }
    
    # Add results for each method
    for method in ['GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']:
        if method in results:
            row_data[f"{method}_estimate"] = results[method]['estimate']
            row_data[f"MWF_{method}"] = results[method]['mwf']
            row_data[f"Lam_{method}"] = results[method]['lambda']
            row_data[f"{method}_Flag_Val"] = results[method]['flag']
        else:
            # Fill with defaults if method failed
            row_data[f"{method}_estimate"] = np.zeros(n_t2)
            row_data[f"MWF_{method}"] = 0.0
            row_data[f"Lam_{method}"] = 0.0
            row_data[f"{method}_Flag_Val"] = 1
    
    return pd.DataFrame([row_data])


class BrainDataProcessor:
    """Main class for brain data processing and analysis."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
        self.matlab_engine = self._setup_matlab()
        # Initialize data arrays
        self.brain_data = None
        self.snr_map = None
        self.mask = None
        self.A = None  # Forward matrix
        self.te = None
        self.t2 = None
        self.dt = None
        self.lambda_range = None
        self.myelin_idx = None
        self.uniform_noise_val = None
        # Results storage
        self.results_list = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.config.log_dir, 'optimized_brain_analysis_parallel.log')
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_matlab(self):
        """Setup MATLAB engine if available."""
        try:
            eng = matlab.engine.start_matlab()
            eng.addpath(self.config.matlab_path, nargout=0)
            self.logger.info("MATLAB engine initialized successfully")
            return eng
        except Exception as e:
            self.logger.warning(f"MATLAB engine not available: {e}")
            return None
    
    def load_data(self):
        """Load all required data files."""
        self.logger.info("Loading data files...")
        
        # Load brain data
        brain_mat = scipy.io.loadmat(self.config.brain_data_path)
        self.brain_data = brain_mat["final_data_2"]
        
        # Load SNR map
        snr_mat = scipy.io.loadmat(self.config.snr_map_path)
        self.snr_map = snr_mat["SNR_MAP"]
        self.snr_map = np.where(self.snr_map == 0, 1, self.snr_map)  # Avoid division by zero
        
        # Load mask
        mask_mat = scipy.io.loadmat(self.config.mask_path)
        self.mask = mask_mat["new_BW"]
        
        self.logger.info(f"Brain data shape: {self.brain_data.shape}")
        self.logger.info(f"SNR map shape: {self.snr_map.shape}")
        self.logger.info(f"Mask shape: {self.mask.shape}")
    
    def setup_parameters(self):
        """Setup analysis parameters and forward matrix."""
        # Time parameters
        self.te = self.config.dte * np.linspace(1, self.config.n_te, self.config.n_te)
        self.t2 = np.linspace(self.config.t2_min, self.config.t2_max, self.config.n_t2)
        self.dt = self.t2[1] - self.t2[0]
        
        # Forward matrix
        self.A = self._create_forward_matrix()
        
        # Lambda range for regularization
        self.lambda_range = np.logspace(
            np.log10(self.config.lambda_min), 
            np.log10(self.config.lambda_max), 
            self.config.n_lambda
        ).reshape(-1, 1)
        
        # Myelin indices
        self.myelin_idx = np.where(self.t2 <= self.config.myelin_threshold)
        
        self.logger.info(f"TE range: {self.te[0]:.1f} - {self.te[-1]:.1f} ms")
        self.logger.info(f"T2 range: {self.t2[0]:.1f} - {self.t2[-1]:.1f} ms")
        self.logger.info(f"Forward matrix shape: {self.A.shape}")
    
    def _create_forward_matrix(self) -> np.ndarray:
        """Create the forward matrix A for the inverse problem."""
        A = np.zeros((self.config.n_te, self.config.n_t2))
        for i in range(self.config.n_te):
            for j in range(self.config.n_t2):
                A[i, j] = np.exp(-self.te[i] / self.t2[j]) * self.dt
        return A
    
    def setup_uniform_noise(self):
        """Setup uniform noise based on representative brain signals."""
        if not self.config.uniform_noise:
            return
            
        self.logger.info("Setting up uniform noise...")
        
        # Get representative coordinates
        coord_pairs = self._get_noise_coordinates()
        
        # Extract signals
        signals = np.array([
            self.brain_data[x, y, :] for x, y in coord_pairs
        ])
        
        # Compute mean signal and normalize
        mean_signal = np.mean(signals, axis=0)
        sol = nnls(self.A, mean_signal)[0]
        factor = np.sum(sol) * self.dt
        mean_signal = mean_signal / factor
        
        # Compute noise from signal tail
        tail = mean_signal[-self.config.tail_length:]
        tail_std = np.std(tail)
        
        # Generate uniform noise
        self.uniform_noise_val = np.random.normal(0, tail_std, size=self.config.n_te)
        
        self.logger.info(f"Uniform noise setup complete. Std: {tail_std:.6f}")
    
    def _get_noise_coordinates(self) -> List[Tuple[int, int]]:
        """Get coordinates for noise estimation."""
        coord_pairs = set()
        x_min, x_max = self.config.noise_coords_x_range
        y_min, y_max = self.config.noise_coords_y_range
        
        attempts = 0
        max_attempts = self.config.num_noise_signals * 10
        
        while len(coord_pairs) < self.config.num_noise_signals and attempts < max_attempts:
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            if self.mask[x, y] == 0:  # Valid voxel
                coord_pairs.add((x, y))
            attempts += 1
            
        return list(coord_pairs)

    def analyze_data_before_processing(self):
        """Analyze the data to predict how many voxels will be processed."""
        self.logger.info("Analyzing data characteristics before processing...")
        
        p, q, _ = self.brain_data.shape
        total_voxels = p * q
        
        # Count voxels above threshold
        above_threshold = np.sum(self.brain_data[:, :, 0] >= self.config.data_threshold)
        below_threshold = total_voxels - above_threshold
        
        # Analyze mask if available
        if self.mask is not None:
            masked_out = np.sum(self.mask == 1)  # Assuming 1 means masked out
            valid_mask = total_voxels - masked_out
        else:
            masked_out = 0
            valid_mask = total_voxels
        
        # Data statistics
        data_min = np.min(self.brain_data[:, :, 0])
        data_max = np.max(self.brain_data[:, :, 0])
        data_mean = np.mean(self.brain_data[:, :, 0])
        data_std = np.std(self.brain_data[:, :, 0])
        
        # Log comprehensive statistics
        self.logger.info("="*50)
        self.logger.info("PRE-PROCESSING DATA ANALYSIS")
        self.logger.info("="*50)
        self.logger.info(f"Brain data shape: {self.brain_data.shape}")
        self.logger.info(f"Total voxels: {total_voxels}")
        self.logger.info(f"Data threshold: {self.config.data_threshold}")
        self.logger.info(f"Voxels above threshold: {above_threshold} ({above_threshold/total_voxels*100:.1f}%)")
        self.logger.info(f"Voxels below threshold: {below_threshold} ({below_threshold/total_voxels*100:.1f}%)")
        
        if self.mask is not None:
            self.logger.info(f"Masked out voxels: {masked_out} ({masked_out/total_voxels*100:.1f}%)")
            self.logger.info(f"Valid mask voxels: {valid_mask} ({valid_mask/total_voxels*100:.1f}%)")
        
        self.logger.info(f"First echo data stats:")
        self.logger.info(f"  Min: {data_min:.2f}")
        self.logger.info(f"  Max: {data_max:.2f}")
        self.logger.info(f"  Mean: {data_mean:.2f}")
        self.logger.info(f"  Std: {data_std:.2f}")
        self.logger.info("="*50)
        
        return {
            'total_voxels': total_voxels,
            'above_threshold': above_threshold,
            'below_threshold': below_threshold,
            'masked_out': masked_out,
            'valid_mask': valid_mask,
            'data_stats': {
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'std': data_std
            }
        }

    def check_zero_mwf(self, result_df: pd.DataFrame) -> bool:
        """
        Check if ANY MWF value is zero or close to zero.
        Returns True if ANY method has zero/low MWF (including reference).
        """
        methods = ['Ref', 'GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        
        # Check if ANY method MWF is below threshold
        for method in methods:
            mwf_col = f"MWF_{method}"
            if mwf_col in result_df.columns:
                mwf_val = result_df[mwf_col].iloc[0]
                if mwf_val <= self.config.mwf_zero_threshold:
                    return True  # Found at least one zero/low MWF
        
        return False  # All MWFs are above threshold

    def _prepare_voxel_chunks(self):
        """Prepare voxel coordinates in chunks for parallel processing."""
        p, q, _ = self.brain_data.shape
        
        # Get all valid voxel coordinates (above threshold)
        valid_coords = []
        for x in range(p):
            for y in range(q):
                if self.brain_data[x, y, 0] >= self.config.data_threshold:
                    valid_coords.append((x, y))
        
        # Split into chunks
        chunks = []
        chunk_size = self.config.chunk_size
        
        for i in range(0, len(valid_coords), chunk_size):
            chunk_coords = valid_coords[i:i + chunk_size]
            
            # Prepare data for this chunk
            chunk_data = (
                chunk_coords,
                self.brain_data,
                self.snr_map,
                self.config.__dict__,  # Convert config to dict for serialization
                self.A,
                self.te,
                self.t2,
                self.dt,
                self.lambda_range,
                self.myelin_idx,
                self.uniform_noise_val
            )
            chunks.append(chunk_data)
        
        return chunks, len(valid_coords)

    def _log_mwf_statistics_by_method(self):
        """Log detailed MWF statistics for each method."""
        if not self.results_list:
            return
        
        methods = ['Ref', 'GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        
        # Combine all results to analyze
        if len(self.results_list) > 0:
            temp_df = pd.concat(self.results_list, ignore_index=True)
            
            for method in methods:
                mwf_col = f"MWF_{method}"
                if mwf_col in temp_df.columns:
                    mwf_values = temp_df[mwf_col].values
                    
                    # Remove any NaN values
                    valid_mwf = mwf_values[~pd.isna(mwf_values)]
                    
                    if len(valid_mwf) > 0:
                        zero_count = np.sum(valid_mwf <= self.config.mwf_zero_threshold)
                        zero_percentage = (zero_count / len(valid_mwf)) * 100
                        
                        self.logger.info(f"  {method}: {zero_count}/{len(valid_mwf)} zero MWF ({zero_percentage:.1f}%)")
                        self.logger.info(f"    Mean MWF: {np.mean(valid_mwf):.4f}, Std: {np.std(valid_mwf):.4f}")
                        self.logger.info(f"    Min: {np.min(valid_mwf):.4f}, Max: {np.max(valid_mwf):.4f}")
                    else:
                        self.logger.info(f"  {method}: No valid MWF values")

    # def run_analysis(self, save_dir: Optional[str] = None):
    #     """Run the complete analysis with parallel processing."""
    #     self.logger.info("Starting parallel brain data analysis...")
        
    #     # Setup
    #     self.load_data()
    #     self.setup_parameters()
    #     self.setup_uniform_noise()
        
    #     # Analyze data before processing
    #     pre_stats = self.analyze_data_before_processing()
        
    #     # Setup parallel processing
    #     n_processes = self.config.n_processes or cpu_count()
    #     self.logger.info(f"Using {n_processes} processes for parallel computation")
        
    #     # Create save directory
    #     if save_dir is None:
    #         date_now = date.today()
    #         day = date_now.strftime('%d')
    #         month = date_now.strftime('%B')[:3]
    #         year = date_now.strftime('%y')
    #         save_dir = f"data/Brain/results_{day}{month}{year}_parallel"
        
    #     save_path = Path(save_dir)
    #     save_path.mkdir(parents=True, exist_ok=True)
        
    #     # Prepare voxel chunks for parallel processing
    #     chunks, total_valid_voxels = self._prepare_voxel_chunks()
    #     p, q, _ = self.brain_data.shape
    #     total_voxels = p * q
        
    #     self.logger.info(f"Total voxels: {total_voxels}")
    #     self.logger.info(f"Valid voxels (above threshold): {total_valid_voxels}")
    #     self.logger.info(f"Processing in {len(chunks)} chunks of size {self.config.chunk_size}")
        
    #     # Process chunks in parallel
    #     processed_count = 0
    #     skipped_count = total_voxels - total_valid_voxels  # Below threshold voxels
    #     zero_mwf_count = 0
    #     processing_errors = 0
        
    #     start_time = time.time()
        
    #     with ProcessPoolExecutor(max_workers=n_processes) as executor:
    #         # Submit all chunks
    #         future_to_chunk = {executor.submit(process_voxel_chunk, chunk): i 
    #                          for i, chunk in enumerate(chunks)}
            
    #         # Process completed chunks
    #         with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
    #             for future in as_completed(future_to_chunk):
    #                 chunk_idx = future_to_chunk[future]
    #                 try:
    #                     chunk_results = future.result()
                        
    #                     # Process results from this chunk
    #                     for result in chunk_results:
    #                         if result is not None:
    #                             # Check if any MWF value is zero or close to zero
    #                             if self.check_zero_mwf(result):
    #                                 zero_mwf_count += 1
                                
    #                             self.results_list.append(result)
    #                             processed_count += 1
    #                         else:
    #                             processing_errors += 1
                        
    #                     # Update progress
    #                     pbar.update(1)
    #                     pbar.set_description(f"Valid: {processed_count}, Zero-MWF: {zero_mwf_count}")
                        
    #                     # Log periodic statistics
    #                     if processed_count > 0 and processed_count % 1000 == 0:
    #                         valid_percentage = (processed_count / total_valid_voxels) * 100
    #                         zero_mwf_percentage = (zero_mwf_count / processed_count) * 100
    #                         self.logger.info(f"Progress: {processed_count}/{total_valid_voxels} valid voxels processed.")
    #                         self.logger.info(f"  Valid results: {processed_count} ({valid_percentage:.1f}%)")
    #                         self.logger.info(f"  Zero/Low MWF: {zero_mwf_count} ({zero_mwf_percentage:.1f}% of valid)")
                        
    #                     # Periodic saving
    #                     if len(self.results_list) >= self.config.checkpoint_interval:
    #                         self._save_checkpoint(save_path, processed_count)
                            
    #                 except Exception as exc:
    #                     self.logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")
    #                     processing_errors += 1
    def run_analysis(self, save_dir: Optional[str] = None):
        """Run the complete analysis with parallel processing."""
        self.logger.info("Starting parallel brain data analysis...")
        
        # Setup
        self.load_data()
        self.setup_parameters()
        self.setup_uniform_noise()
        
        # Analyze data before processing
        pre_stats = self.analyze_data_before_processing()
        
        # Setup parallel processing
        n_processes = self.config.n_processes or cpu_count()
        self.logger.info(f"Using {n_processes} processes for parallel computation")
        
        # Create save directory
        if save_dir is None:
            date_now = date.today()
            day = date_now.strftime('%d')
            month = date_now.strftime('%B')[:3]
            year = date_now.strftime('%y')
            save_dir = f"data/Brain/results_{day}{month}{year}_parallel"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare voxel chunks for parallel processing
        chunks, total_valid_voxels = self._prepare_voxel_chunks()
        p, q, _ = self.brain_data.shape
        total_voxels = p * q
        
        self.logger.info(f"Total voxels: {total_voxels}")
        self.logger.info(f"Valid voxels (above threshold): {total_valid_voxels}")
        self.logger.info(f"Processing in {len(chunks)} chunks of size {self.config.chunk_size}")
        
        # Process chunks in parallel
        processed_count = 0
        skipped_count = total_voxels - total_valid_voxels  # Below threshold voxels
        zero_mwf_count = 0
        processing_errors = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit all chunks
            future_to_chunk = {executor.submit(process_voxel_chunk, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            # Process completed chunks
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_results = future.result()
                        
                        # Process results from this chunk
                        for result in chunk_results:
                            if result is not None:
                                # Check if any MWF value is zero or close to zero
                                if self.check_zero_mwf(result):
                                    zero_mwf_count += 1
                                
                                self.results_list.append(result)
                                processed_count += 1
                            else:
                                processing_errors += 1
                        
                        # Update progress
                        pbar.update(1)
                        pbar.set_description(f"Valid: {processed_count}, Zero-MWF: {zero_mwf_count}")
                        
                        # Log periodic statistics
                        if processed_count > 0 and processed_count % 1000 == 0:
                            valid_percentage = (processed_count / total_valid_voxels) * 100
                            zero_mwf_percentage = (zero_mwf_count / processed_count) * 100
                            self.logger.info(f"Progress: {processed_count}/{total_valid_voxels} valid voxels processed.")
                            self.logger.info(f"  Valid results: {processed_count} ({valid_percentage:.1f}%)")
                            self.logger.info(f"  Zero/Low MWF: {zero_mwf_count} ({zero_mwf_percentage:.1f}% of valid)")
                        
                        # Periodic saving
                        if len(self.results_list) >= self.config.checkpoint_interval:
                            self._save_checkpoint(save_path, processed_count)
                            
                    except Exception as exc:
                        self.logger.error(f"Chunk {chunk_idx} generated an exception: {exc}")
                        processing_errors += 1
        
        # Final save
        self._save_final_results(save_path)
        
        # Final statistics
        elapsed_time = time.time() - start_time
        valid_percentage = (processed_count / total_valid_voxels) * 100 if total_valid_voxels > 0 else 0
        zero_mwf_percentage = (zero_mwf_count / processed_count) * 100 if processed_count > 0 else 0
        processing_error_percentage = (processing_errors / total_valid_voxels) * 100 if total_valid_voxels > 0 else 0
        below_threshold_percentage = (skipped_count / total_voxels) * 100 if total_voxels > 0 else 0
        
        self.logger.info("="*70)
        self.logger.info("PARALLEL ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
        self.logger.info("="*70)
        self.logger.info(f"Total voxels in dataset: {total_voxels}")
        self.logger.info(f"Valid voxels (above threshold): {total_valid_voxels}")
        self.logger.info(f"Processing time: {elapsed_time/60:.2f} minutes")
        self.logger.info(f"Average time per valid voxel: {elapsed_time/total_valid_voxels:.4f} seconds")
        self.logger.info(f"Parallel processing with {n_processes} processes")
        self.logger.info(f"Processed in {len(chunks)} chunks")
        self.logger.info("")
        self.logger.info("PROCESSING RESULTS:")
        self.logger.info(f"  Valid results: {processed_count} ({valid_percentage:.2f}%)")
        self.logger.info(f"  Processing errors: {processing_errors} ({processing_error_percentage:.2f}%)")
        self.logger.info(f"  Below threshold: {skipped_count} ({below_threshold_percentage:.2f}%)")
        self.logger.info("")
        self.logger.info("MWF ANALYSIS (for valid results):")
        self.logger.info(f"  Voxels with all MWF ≈ 0: {zero_mwf_count} ({zero_mwf_percentage:.2f}% of valid)")
        self.logger.info(f"  Voxels with meaningful MWF: {processed_count - zero_mwf_count} ({100-zero_mwf_percentage:.2f}% of valid)")
        self.logger.info("")
        self.logger.info("PERFORMANCE METRICS:")
        if processed_count > 0:
            self.logger.info(f"  Average time per valid result: {elapsed_time/processed_count:.4f} seconds")
            meaningful_results = processed_count - zero_mwf_count
            if meaningful_results > 0:
                self.logger.info(f"  Average time per meaningful result: {elapsed_time/meaningful_results:.4f} seconds")
        self.logger.info("")
        self.logger.info("DATA QUALITY ASSESSMENT:")
        usable_percentage = ((processed_count - zero_mwf_count) / total_voxels) * 100 if total_voxels > 0 else 0
        self.logger.info(f"  Overall usable data rate: {usable_percentage:.2f}% of total voxels")
        if total_valid_voxels > 0:
            success_rate = (processed_count / total_valid_voxels) * 100
            self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
        self.logger.info("="*70)
        
        # Additional detailed breakdown by method
        if processed_count > 0:
            self.logger.info("")
            self.logger.info("MWF STATISTICS BY METHOD:")
            self._log_mwf_statistics_by_method()
            self.logger.info("="*70)

    def _prepare_voxel_chunks(self):
        """Prepare voxel data in chunks for parallel processing."""
        p, q, _ = self.brain_data.shape
        chunks = []
        total_valid_voxels = 0
        
        # Create chunks of voxel coordinates and corresponding data
        chunk_coords = []
        chunk_brain_data = []
        chunk_snr_data = []
        
        for x in range(p):
            for y in range(q):
                if self.brain_data[x, y, 0] >= self.config.data_threshold:
                    chunk_coords.append((x, y))
                    chunk_brain_data.append(self.brain_data[x, y, :])
                    chunk_snr_data.append(self.snr_map[x, y])
                    total_valid_voxels += 1
                    
                    # When chunk is full, create a chunk package
                    if len(chunk_coords) >= self.config.chunk_size:
                        chunk_data = (
                            chunk_coords.copy(),
                            np.array(chunk_brain_data),
                            np.array(chunk_snr_data),
                            self.A,
                            self.lambda_range,
                            self.uniform_noise_val,
                            self.config
                        )
                        chunks.append(chunk_data)
                        
                        # Reset for next chunk
                        chunk_coords = []
                        chunk_brain_data = []
                        chunk_snr_data = []
        
        # Handle remaining voxels
        if chunk_coords:
            chunk_data = (
                chunk_coords,
                np.array(chunk_brain_data),
                np.array(chunk_snr_data),
                self.A,
                self.lambda_range,
                self.uniform_noise_val,
                self.config
            )
            chunks.append(chunk_data)
        
        return chunks, total_valid_voxels

    def _save_checkpoint(self, save_path: Path, count: int):
        """Save intermediate results."""
        if self.results_list:
            df = pd.concat(self.results_list, ignore_index=True)
            checkpoint_file = save_path / f"checkpoint_{count}.pkl"
            df.to_pickle(checkpoint_file)
            self.logger.info(f"Saved checkpoint with {len(df)} results")
            self.results_list.clear()
    
    def _save_final_results(self, save_path: Path):
        """Save final results."""
        # Collect all checkpoint files
        checkpoint_files = list(save_path.glob("checkpoint_*.pkl"))
        all_dfs = []
        
        # Load checkpoint files
        for file in checkpoint_files:
            df = pd.read_pickle(file)
            all_dfs.append(df)
            file.unlink()  # Remove checkpoint file
        
        # Add remaining results
        if self.results_list:
            df = pd.concat(self.results_list, ignore_index=True)
            all_dfs.append(df)
        
        # Combine and save
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            # Generate filename
            date_now = date.today()
            day = date_now.strftime('%d')
            month = date_now.strftime('%B')[:3]
            year = date_now.strftime('%y')
            
            p, q, _ = self.brain_data.shape
            filename = f"est_table_xcoordlen_{p}_ycoordlen_{q}_optimized_parallel_{day}{month}{year}.pkl"
            
            final_path = save_path / filename
            final_df.to_pickle(final_path)
            
            self.logger.info(f"Final results saved to {final_path}")
            self.logger.info(f"Total results: {len(final_df)}")

    def _log_mwf_statistics_by_method(self):
        """Log detailed MWF statistics for each method."""
        if not self.results_list:
            return
        
        methods = ['Ref', 'GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        mwf_threshold = 1e-3
        
        # Combine all results to analyze
        if len(self.results_list) > 0:
            temp_df = pd.concat(self.results_list, ignore_index=True)
            
            for method in methods:
                mwf_col = f"MWF_{method}"
                if mwf_col in temp_df.columns:
                    mwf_values = temp_df[mwf_col].values
                    
                    # Remove any NaN values
                    valid_mwf = mwf_values[~pd.isna(mwf_values)]
                    
                    if len(valid_mwf) > 0:
                        zero_count = np.sum(valid_mwf <= mwf_threshold)
                        zero_percentage = (zero_count / len(valid_mwf)) * 100
                        
                        self.logger.info(f"  {method}: {zero_count}/{len(valid_mwf)} zero MWF ({zero_percentage:.1f}%)")
                        self.logger.info(f"    Mean MWF: {np.mean(valid_mwf):.4f}, Std: {np.std(valid_mwf):.4f}")
                        self.logger.info(f"    Min: {np.min(valid_mwf):.4f}, Max: {np.max(valid_mwf):.4f}")
                    else:
                        self.logger.info(f"  {method}: No valid MWF values")

    def check_zero_mwf(self, result_df: pd.DataFrame, mwf_threshold: float = 1e-6) -> bool:
        """Check if all MWF values are zero or close to zero."""
        methods = ['GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        
        # Check if all method MWFs are below threshold
        all_zero = True
        for method in methods:
            mwf_col = f"MWF_{method}"
            if mwf_col in result_df.columns:
                mwf_val = result_df[mwf_col].iloc[0]
                if mwf_val > mwf_threshold:
                    all_zero = False
                    break
        
        # Also check reference MWF
        if 'MWF_Ref' in result_df.columns:
            ref_mwf = result_df['MWF_Ref'].iloc[0]
            if ref_mwf > mwf_threshold:
                all_zero = False
        
        return all_zero

def main():
    """Main execution function."""
    # Create configuration
    config = Config()
    
    # Override any config parameters as needed
    # config.uniform_noise = False  # Example override
    
    # Create processor and run analysis
    processor = BrainDataProcessor(config)
    processor.run_analysis()


if __name__ == "__main__":
    main()
    

# import os
# import sys
# sys.path.append(".")
# import logging
# import pickle
# import time
# from datetime import date
# from pathlib import Path
# from dataclasses import dataclass
# from typing import Tuple, List, Optional, Dict, Any
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Set environment variables before any MOSEK-related imports
# os.environ["MOSEKLM_LICENSE_FILE"] = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import logging
# logging.info(f'MOSEK License Set from {os.environ["MOSEKLM_LICENSE_FILE"]}')

# # Now proceed with other imports
# import numpy as np
# import pandas as pd
# import scipy.io
# import matplotlib.pyplot as plt
# from scipy.stats import wasserstein_distance
# from scipy.linalg import norm as linalg_norm
# from scipy.optimize import nnls
# from tqdm import tqdm
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Simulations.LRalgo import LocReg_Ito_mod, LocReg_Ito_mod_deriv, LocReg_Ito_mod_deriv2
# from Simulations.upenzama import UPEN_Zama
# import matlab.engine
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support
# import mosek

# @dataclass
# class Config:
#     """Configuration parameters for the analysis."""
#     # Data paths
#     brain_data_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat"
#     snr_map_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat"
#     mask_path: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"
#     log_dir: str = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1"
#     matlab_path: str = r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test'

#     # mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
#     # os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
#     # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#     # logging.info(f'MOSEK License Set from {mosek_license_path}')
#     # Analysis parameters
#     dte: float = 11.3
#     n_te: int = 32
#     n_t2: int = 150
#     t2_min: float = 10.0
#     t2_max: float = 200.0
#     myelin_threshold: float = 40.0
#     data_threshold: float = 50.0
    
#     # Regularization parameters
#     lambda_min: float = 1e-4
#     lambda_max: float = 1e-1
#     n_lambda: int = 10
#     gamma_init: float = 0.5
#     max_iter: int = 50
    
#     # Noise parameters
#     uniform_noise: bool = True
#     num_noise_signals: int = 1000
#     noise_coords_x_range: Tuple[int, int] = (130, 200)
#     noise_coords_y_range: Tuple[int, int] = (100, 200)
#     tail_length: int = 3
    
#     # Processing parameters
#     checkpoint_interval: int = 1000
#     checkpoint_time_interval: int = 900


# def voxel_worker_wrapper(args):
#     self_instance, x, y = args
#     return self_instance._voxel_worker(x, y)

# class BrainDataProcessor:
#     """Main class for brain data processing and analysis."""
#     def __init__(self, config: Config):
#         self.config = config
#         self.logger = self._setup_logging()
#         # self.matlab_engine = self._setup_matlab()
#         # Initialize data arrays
#         self.brain_data = None
#         self.snr_map = None
#         self.mask = None
#         self.A = None  # Forward matrix
#         self.te = None
#         self.t2 = None
#         self.dt = None
#         self.lambda_range = None
#         self.myelin_idx = None
#         self.uniform_noise_val = None
#         # Results storage
#         self.results_list = []
        
#     def _setup_logging(self) -> logging.Logger:
#         """Setup logging configuration."""
#         os.makedirs(self.config.log_dir, exist_ok=True)
        
#         logger = logging.getLogger(__name__)
#         logger.setLevel(logging.INFO)
#         logger.handlers.clear()
        
#         # File handler
#         file_handler = logging.FileHandler(
#             os.path.join(self.config.log_dir, 'optimized_brain_analysis.log')
#         )
#         file_handler.setLevel(logging.INFO)
        
#         # Console handler
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
        
#         # Formatter
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
        
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)
        
#         return logger
    
#     def _setup_matlab(self):
#         """Setup MATLAB engine if available."""
#         try:
#             eng = matlab.engine.start_matlab()
#             eng.addpath(self.config.matlab_path, nargout=0)
#             self.logger.info("MATLAB engine initialized successfully")
#             return eng
#         except Exception as e:
#             self.logger.warning(f"MATLAB engine not available: {e}")
#             return None
    
#     def load_data(self):
#         """Load all required data files."""
#         self.logger.info("Loading data files...")
        
#         # Load brain data
#         brain_mat = scipy.io.loadmat(self.config.brain_data_path)
#         self.brain_data = brain_mat["final_data_2"]
        
#         # Load SNR map
#         snr_mat = scipy.io.loadmat(self.config.snr_map_path)
#         self.snr_map = snr_mat["SNR_MAP"]
#         self.snr_map = np.where(self.snr_map == 0, 1, self.snr_map)  # Avoid division by zero
        
#         # Load mask
#         mask_mat = scipy.io.loadmat(self.config.mask_path)
#         self.mask = mask_mat["new_BW"]
        
#         self.logger.info(f"Brain data shape: {self.brain_data.shape}")
#         self.logger.info(f"SNR map shape: {self.snr_map.shape}")
#         self.logger.info(f"Mask shape: {self.mask.shape}")
    
#     def setup_parameters(self):
#         """Setup analysis parameters and forward matrix."""
#         # Time parameters
#         self.te = self.config.dte * np.linspace(1, self.config.n_te, self.config.n_te)
#         self.t2 = np.linspace(self.config.t2_min, self.config.t2_max, self.config.n_t2)
#         self.dt = self.t2[1] - self.t2[0]
        
#         # Forward matrix
#         self.A = self._create_forward_matrix()
        
#         # Lambda range for regularization
#         self.lambda_range = np.logspace(
#             np.log10(self.config.lambda_min), 
#             np.log10(self.config.lambda_max), 
#             self.config.n_lambda
#         ).reshape(-1, 1)
        
#         # Myelin indices
#         self.myelin_idx = np.where(self.t2 <= self.config.myelin_threshold)
        
#         self.logger.info(f"TE range: {self.te[0]:.1f} - {self.te[-1]:.1f} ms")
#         self.logger.info(f"T2 range: {self.t2[0]:.1f} - {self.t2[-1]:.1f} ms")
#         self.logger.info(f"Forward matrix shape: {self.A.shape}")
    
#     def _create_forward_matrix(self) -> np.ndarray:
#         """Create the forward matrix A for the inverse problem."""
#         A = np.zeros((self.config.n_te, self.config.n_t2))
#         for i in range(self.config.n_te):
#             for j in range(self.config.n_t2):
#                 A[i, j] = np.exp(-self.te[i] / self.t2[j]) * self.dt
#         return A
    
#     def setup_uniform_noise(self):
#         """Setup uniform noise based on representative brain signals."""
#         if not self.config.uniform_noise:
#             return
            
#         self.logger.info("Setting up uniform noise...")
        
#         # Get representative coordinates
#         coord_pairs = self._get_noise_coordinates()
        
#         # Extract signals
#         signals = np.array([
#             self.brain_data[x, y, :] for x, y in coord_pairs
#         ])
        
#         # Compute mean signal and normalize
#         mean_signal = np.mean(signals, axis=0)
#         sol = nnls(self.A, mean_signal)[0]
#         factor = np.sum(sol) * self.dt
#         mean_signal = mean_signal / factor
        
#         # Compute noise from signal tail
#         tail = mean_signal[-self.config.tail_length:]
#         tail_std = np.std(tail)
        
#         # Generate uniform noise
#         self.uniform_noise_val = np.random.normal(0, tail_std, size=self.config.n_te)
        
#         self.logger.info(f"Uniform noise setup complete. Std: {tail_std:.6f}")
    
#     def _get_noise_coordinates(self) -> List[Tuple[int, int]]:
#         """Get coordinates for noise estimation."""
#         coord_pairs = set()
#         x_min, x_max = self.config.noise_coords_x_range
#         y_min, y_max = self.config.noise_coords_y_range
        
#         attempts = 0
#         max_attempts = self.config.num_noise_signals * 10
        
#         while len(coord_pairs) < self.config.num_noise_signals and attempts < max_attempts:
#             x = np.random.randint(x_min, x_max)
#             y = np.random.randint(y_min, y_max)
#             if self.mask[x, y] == 0:  # Valid voxel
#                 coord_pairs.add((x, y))
#             attempts += 1
            
#         return list(coord_pairs)
    
#     # def process_voxel(self, x: int, y: int) -> Optional[pd.DataFrame]:
#     #     """Process a single voxel and return results."""
#     #     # Check data threshold
#     #     if self.brain_data[x,y, 0] < self.config.data_threshold:
#     #         return None
#     #     try:
#     #         #check 128, 154, 140, 154
#     #         curr_data = self.brain_data[x,y, :].copy()
#     #         curr_snr = self.snr_map[x,y]
            
#     #         # Normalize data
#     #         curr_data = self._normalize_data(curr_data)
            
#     #         # Get reference solution (GCV without noise)
#     #         ref_rec, ref_lamb = GCV_NNLS(curr_data, self.A,self.lambda_range)
#     #         ref_rec = ref_rec[:, 0]
#     #         ref_lamb = np.squeeze(ref_lamb)
#     #         ref_rec, ref_mwf, ref_flag = self._compute_mwf_filtered(ref_rec)
#     #         self.logger.info(f"pixel {x} and {y} with ref_mwf {ref_mwf}")
#     #         # Add noise
#     #         data_gt = self.A @ ref_rec
#     #         noisy_data = data_gt + self.uniform_noise_val
            
#     #         # Apply regularization methods
#     #         results = self._apply_regularization_methods(noisy_data, ref_lamb)
            
#     #         # Create result dataframe
#     #         return self._create_result_dataframe(
#     #             x, y, noisy_data, curr_snr, ref_rec, ref_mwf, ref_lamb, ref_flag, results
#     #         )
            
#     #     except Exception as e:
#     #         self.logger.error(f"Error processing voxel ({x}, {y}): {e}")
#     #         return None
    
#     def _normalize_data(self, data: np.ndarray) -> np.ndarray:
#         """Normalize data using NNLS solution."""
#         # sol = nonnegtik_hnorm(self.A, data, 0, '0', nargin=4)[0]
#         sol = nnls(self.A, data)[0]
#         factor = np.sum(sol) * self.dt
#         return data / factor if factor > 0 else data
    
#     def _apply_regularization_methods(self, data: np.ndarray, ref_lamb: float) -> Dict[str, Any]:
#         """Apply all regularization methods to the data."""
#         results = {}
        
#         # GCV
#         gcv_rec, gcv_lamb = GCV_NNLS(data, self.A, self.lambda_range)
#         gcv_rec = gcv_rec[:, 0]
#         gcv_lamb = np.squeeze(gcv_lamb)
#         results['GCV'] = {
#             'estimate': gcv_rec,
#             'lambda': gcv_lamb,
#             'mwf': None,
#             'flag': None
#         }
        
#         # LocReg variants
#         ini_lamb = gcv_lamb
        
#         # Standard LocReg
#         lr_rec, lr_lamb, _, _, _ = LocReg_Ito_mod(
#             data, self.A, ini_lamb, self.config.gamma_init, maxiter=self.config.max_iter
#         )
#         results['LR'] = {
#             'estimate': lr_rec,
#             'lambda': lr_lamb,
#             'mwf': None,
#             'flag': None
#         }
        
#         # LocReg with 1st derivative
#         lr1d_rec, lr1d_lamb, _, _, _ = LocReg_Ito_mod_deriv(
#             data, self.A, ini_lamb, self.config.gamma_init, maxiter=self.config.max_iter
#         )
#         results['LR1D'] = {
#             'estimate': lr1d_rec,
#             'lambda': lr1d_lamb,
#             'mwf': None,
#             'flag': None
#         }
        
#         # LocReg with 2nd derivative
#         lr2d_rec, lr2d_lamb, _, _, _ = LocReg_Ito_mod_deriv2(
#             data, self.A, ini_lamb, self.config.gamma_init, maxiter=self.config.max_iter
#         )
#         results['LR2D'] = {
#             'estimate': lr2d_rec,
#             'lambda': lr2d_lamb,
#             'mwf': None,
#             'flag': None
#         }
        
#         # # UPEN
#         # std_dev = np.std(data[-5:])
#         # snr_est = np.max(np.abs(data)) / std_dev
#         # threshold = 1.05 * np.sqrt(self.A.shape[0]) * np.max(data) / snr_est
        
#         # upen_rec, upen_lamb = UPEN_Zama(
#         #     self.A, data, gcv_rec, threshold, 1e-3, 50, 1e-5
#         # )
#         # results['UPEN'] = {
#         #     'estimate': upen_rec,
#         #     'lambda': upen_lamb,
#         #     'mwf': None,
#         #     'flag': None
#         # }
        
#         # Compute MWF and flags for all methods
#         for method in results:
#             rec, mwf, flag = self._compute_mwf_filtered(results[method]['estimate'])
#             results[method]['estimate'] = rec
#             results[method]['mwf'] = mwf
#             results[method]['flag'] = flag
#             self.logger.info(f"{method} MWF {mwf}")

#         return results
    
#     def _compute_mwf_filtered(self, reconstruction: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, float, int]:
#         """Compute MWF with filtering for edge cases."""
#         if np.all(np.abs(reconstruction) < tol) or np.all(reconstruction[:-1] == 0):
#             return reconstruction, 0.0, 1
            
#         try:
#             total_sum = np.sum(reconstruction) * self.dt
#             if total_sum == 0:
#                 return reconstruction, 0.0, 1
                
#             f_normalized = reconstruction / total_sum
#             total_mwf = np.cumsum(f_normalized)
#             mwf = total_mwf[self.myelin_idx[0][-1]]
#             return f_normalized, mwf, 0
            
#         except (ZeroDivisionError, IndexError):
#             return reconstruction, 0.0, 1
    
#     def _create_result_dataframe(self, x: int, y: int, data: np.ndarray, snr: float,
#                                ref_rec: np.ndarray, ref_mwf: float, ref_lamb: float,
#                                ref_flag: int, results: Dict[str, Any]) -> pd.DataFrame:
#         """Create result dataframe for a single voxel."""
#         row_data = {
#             "X_val": x,
#             "Y_val": y,
#             "curr_data": data,
#             "noise": self.uniform_noise_val,
#             "curr_SNR": snr,
#             "ref_estimate": ref_rec,
#             "MWF_Ref": ref_mwf,
#             "Lam_Ref": ref_lamb,
#             "Ref_Flag_Val": ref_flag
#         }
        
        
#         # Add results for each method
#         for method in ['GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']:
#             if method in results:
#                 row_data[f"{method}_estimate"] = results[method]['estimate']
#                 row_data[f"MWF_{method}"] = results[method]['mwf']
#                 row_data[f"Lam_{method}"] = results[method]['lambda']
#                 row_data[f"{method}_Flag_Val"] = results[method]['flag']
#             else:
#                 # Fill with defaults if method failed
#                 row_data[f"{method}_estimate"] = np.zeros(self.config.n_t2)
#                 row_data[f"MWF_{method}"] = 0.0
#                 row_data[f"Lam_{method}"] = 0.0
#                 row_data[f"{method}_Flag_Val"] = 1
        
#         return pd.DataFrame([row_data])





















    
    # def run_analysis(self, save_dir: Optional[str] = None):
    #     """Run the complete analysis."""
    #     self.logger.info("Starting brain data analysis...")
        
    #     # Setup
    #     self.load_data()
    #     self.setup_parameters()
    #     self.setup_uniform_noise()
        
    #     # Create save directory
    #     if save_dir is None:
    #         date_now = date.today()
    #         day = date_now.strftime('%d')
    #         month = date_now.strftime('%B')[:3]
    #         year = date_now.strftime('%y')
    #         save_dir = f"data/Brain/results_{day}{month}{year}"
        
    #     save_path = Path(save_dir)
    #     save_path.mkdir(parents=True, exist_ok=True)
        
    #     # Process all voxels
    #     p, q, _ = self.brain_data.shape
    #     total_voxels = p * q
    #     processed_count = 0
        
    #     start_time = time.time()
        
    #     with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
    #         for x in range(p):
    #             for y in range(q):
    #                 result = self.process_voxel(x, y)
    #                 if result is not None:
    #                     self.results_list.append(result)
    #                     processed_count += 1
                    
    #                 pbar.update(1)
                    
    #                 # Periodic saving
    #                 if len(self.results_list) >= self.config.checkpoint_interval:
    #                     self._save_checkpoint(save_path, processed_count)
        
    #     # Final save
    #     self._save_final_results(save_path)
        
    #     elapsed_time = time.time() - start_time
    #     self.logger.info(f"Analysis complete. Processed {processed_count} voxels in {elapsed_time/60:.2f} minutes")

    def analyze_data_before_processing(self):
        """Analyze the data to predict how many voxels will be processed."""
        self.logger.info("Analyzing data characteristics before processing...")
        
        p, q, _ = self.brain_data.shape
        total_voxels = p * q
        
        # Count voxels above threshold
        above_threshold = np.sum(self.brain_data[:, :, 0] >= self.config.data_threshold)
        below_threshold = total_voxels - above_threshold
        
        # Analyze mask if available
        if self.mask is not None:
            masked_out = np.sum(self.mask == 1)  # Assuming 1 means masked out
            valid_mask = total_voxels - masked_out
        else:
            masked_out = 0
            valid_mask = total_voxels
        
        # Data statistics
        data_min = np.min(self.brain_data[:, :, 0])
        data_max = np.max(self.brain_data[:, :, 0])
        data_mean = np.mean(self.brain_data[:, :, 0])
        data_std = np.std(self.brain_data[:, :, 0])
        
        # Log comprehensive statistics
        self.logger.info("="*50)
        self.logger.info("PRE-PROCESSING DATA ANALYSIS")
        self.logger.info("="*50)
        self.logger.info(f"Brain data shape: {self.brain_data.shape}")
        self.logger.info(f"Total voxels: {total_voxels}")
        self.logger.info(f"Data threshold: {self.config.data_threshold}")
        self.logger.info(f"Voxels above threshold: {above_threshold} ({above_threshold/total_voxels*100:.1f}%)")
        self.logger.info(f"Voxels below threshold: {below_threshold} ({below_threshold/total_voxels*100:.1f}%)")
        
        if self.mask is not None:
            self.logger.info(f"Masked out voxels: {masked_out} ({masked_out/total_voxels*100:.1f}%)")
            self.logger.info(f"Valid mask voxels: {valid_mask} ({valid_mask/total_voxels*100:.1f}%)")
        
        self.logger.info(f"First echo data stats:")
        self.logger.info(f"  Min: {data_min:.2f}")
        self.logger.info(f"  Max: {data_max:.2f}")
        self.logger.info(f"  Mean: {data_mean:.2f}")
        self.logger.info(f"  Std: {data_std:.2f}")
        self.logger.info("="*50)
        
        return {
            'total_voxels': total_voxels,
            'above_threshold': above_threshold,
            'below_threshold': below_threshold,
            'masked_out': masked_out,
            'valid_mask': valid_mask,
            'data_stats': {
                'min': data_min,
                'max': data_max,
                'mean': data_mean,
                'std': data_std
            }
        }

    def get_processing_statistics(self) -> Dict[str, int]:
        """Get detailed statistics about why voxels were skipped."""
        p, q, _ = self.brain_data.shape
        stats = {
            'total_voxels': p * q,
            'below_threshold': 0,
            'processing_errors': 0,
            'valid_results': 0
        }
        
        for x in range(p):
            for y in range(q):
                if self.brain_data[x, y, 0] < self.config.data_threshold:
                    stats['below_threshold'] += 1
                else:
                    try:
                        # Quick test - just try to normalize the data
                        curr_data = self.brain_data[x, y, :].copy()
                        self._normalize_data(curr_data)
                        stats['valid_results'] += 1
                    except:
                        stats['processing_errors'] += 1
        
        return stats

    def process_voxel(self, x: int, y: int) -> Optional[pd.DataFrame]:
        """Process a single voxel and return results."""
        # Check data threshold
        if self.brain_data[x, y, 0] < self.config.data_threshold:
            return None
            
        try:
            curr_data = self.brain_data[x, y, :].copy()
            curr_snr = self.snr_map[x, y]
            
            # Normalize data
            curr_data = self._normalize_data(curr_data)
            
            # Get reference solution (GCV without noise)
            ref_rec, ref_lamb = GCV_NNLS(curr_data, self.A, self.lambda_range)
            ref_rec = ref_rec[:, 0]
            ref_lamb = np.squeeze(ref_lamb)
            ref_rec, ref_mwf, ref_flag = self._compute_mwf_filtered(ref_rec)
            self.logger.debug(f"Pixel ({x}, {y}) - ref_mwf: {ref_mwf}")
            
            # Add noise
            data_gt = self.A @ ref_rec
            noisy_data = data_gt + self.uniform_noise_val
            
            # Apply regularization methods
            results = self._apply_regularization_methods(noisy_data, ref_lamb)
            
            # Create result dataframe
            result_df = self._create_result_dataframe(
                x, y, noisy_data, curr_snr, ref_rec, ref_mwf, ref_lamb, ref_flag, results
            )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing voxel ({x}, {y}): {e}")
            return None

    def check_zero_mwf(self, result_df: pd.DataFrame, mwf_threshold: float = 1e-2) -> bool:
        """Check if all MWF values are zero or close to zero."""
        methods = ['GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        
        # Check if all method MWFs are below threshold
        all_zero = True
        for method in methods:
            mwf_col = f"MWF_{method}"
            if mwf_col in result_df.columns:
                mwf_val = result_df[mwf_col].iloc[0]
                if mwf_val > mwf_threshold:
                    all_zero = False
                    break
        
        # Also check reference MWF
        if 'MWF_Ref' in result_df.columns:
            ref_mwf = result_df['MWF_Ref'].iloc[0]
            if ref_mwf > mwf_threshold:
                all_zero = False
        
        return all_zero

    def _log_mwf_statistics_by_method(self):
        """Log detailed MWF statistics for each method."""
        if not self.results_list:
            return
        
        methods = ['Ref', 'GCV', 'LR', 'LR1D', 'LR2D', 'UPEN']
        mwf_threshold = 1e-2
        
        # Combine all results to analyze
        if len(self.results_list) > 0:
            temp_df = pd.concat(self.results_list, ignore_index=True)
            
            for method in methods:
                mwf_col = f"MWF_{method}"
                if mwf_col in temp_df.columns:
                    mwf_values = temp_df[mwf_col].values
                    
                    # Remove any NaN values
                    valid_mwf = mwf_values[~pd.isna(mwf_values)]
                    
                    if len(valid_mwf) > 0:
                        zero_count = np.sum(valid_mwf <= mwf_threshold)
                        zero_percentage = (zero_count / len(valid_mwf)) * 100
                        
                        self.logger.info(f"  {method}: {zero_count}/{len(valid_mwf)} zero MWF ({zero_percentage:.1f}%)")
                        self.logger.info(f"    Mean MWF: {np.mean(valid_mwf):.4f}, Std: {np.std(valid_mwf):.4f}")
                        self.logger.info(f"    Min: {np.min(valid_mwf):.4f}, Max: {np.max(valid_mwf):.4f}")
                    else:
                        self.logger.info(f"  {method}: No valid MWF values")

    def _voxel_worker(self, x, y):
        try:
            # Check early threshold for fast skipping
            if self.brain_data[x, y, 0] < self.config.data_threshold:
                return ('below_threshold', x, y, None)

            result = self.process_voxel(x, y)

            if result is None:
                return ('error', x, y, None)

            # Check for zero MWF values
            if self.check_zero_mwf(result):
                return ('zero_mwf', x, y, result)

            return ('valid', x, y, result)

        except Exception as e:
            self.logger.error(f"Exception in worker for voxel ({x}, {y}): {e}")
            return ('error', x, y, None)

       
    # def run_analysis(self, save_dir: Optional[str] = None, resume_from: Optional[str] = None):
    #     """Run the complete analysis with optional checkpoint resume."""
    #     self.logger.info("Starting brain data analysis...")

    #     # Setup
    #     self.load_data()
    #     self.setup_parameters()
    #     self.setup_uniform_noise()
    #     pre_stats = self.analyze_data_before_processing()

    #     # Create save path
    #     if save_dir is None:
    #         date_now = date.today()
    #         save_dir = f"data/Brain/results_{date_now.strftime('%d%b%y')}"
    #     save_path = Path(save_dir)
    #     save_path.mkdir(parents=True, exist_ok=True)

    #     p, q, _ = self.brain_data.shape
    #     total_voxels = p * q
        
    #     # Handle checkpoint resume
    #     processed_voxels = set()
    #     initial_processed_count = 0
        
    #     if resume_from:
    #         processed_voxels, initial_processed_count = self._load_checkpoint_state(resume_from, save_path)
    #         self.logger.info(f"Resuming from checkpoint: {initial_processed_count} voxels already processed")
        
    #     # Create list of remaining voxels to process
    #     all_voxels = [(self, x, y) for x in range(p) for y in range(q)]
    #     voxel_coords = [v for v in all_voxels if (v[1], v[2]) not in processed_voxels]
    #     remaining_voxels = len(voxel_coords)

    #     # Counters (add to existing counts if resuming)
    #     processed_count = initial_processed_count
    #     zero_mwf_count = 0
    #     skipped_count = 0
    #     below_threshold_count = 0
    #     processing_errors = 0

    #     self.logger.info(f"Total voxels: {total_voxels}")
    #     self.logger.info(f"Already processed: {initial_processed_count}")
    #     self.logger.info(f"Remaining to process: {remaining_voxels}")
    #     self.logger.info(f"Brain data shape: {self.brain_data.shape}")
        
    #     if remaining_voxels == 0:
    #         self.logger.info("All voxels already processed!")
    #         return
        
    #     start_time = time.time()
        
    #     # Process remaining voxels
    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         with tqdm(total=remaining_voxels, desc="Processing remaining voxels") as pbar:
    #             for status, x, y, result in pool.imap_unordered(voxel_worker_wrapper, voxel_coords):
    #                 if status == 'valid':
    #                     self.results_list.append(result)
    #                     processed_count += 1
    #                 elif status == 'zero_mwf':
    #                     self.results_list.append(result)
    #                     processed_count += 1
    #                     zero_mwf_count += 1
    #                 elif status == 'below_threshold':
    #                     skipped_count += 1
    #                     below_threshold_count += 1
    #                 elif status == 'error':
    #                     skipped_count += 1
    #                     processing_errors += 1

    #                 pbar.update(1)

    #                 # Progress reporting (adjust for resume)
    #                 current_new = len([r for r in [status] if r in ['valid', 'zero_mwf']])
    #                 if current_new > 0 and (processed_count - initial_processed_count) % 1000 == 0:
    #                     valid_pct = (processed_count / total_voxels) * 100
    #                     self.logger.info(f"Progress: {processed_count}/{total_voxels} ({valid_pct:.1f}%)")
    #                     self.logger.info(f"  New in this session: {processed_count - initial_processed_count}")
                    
    #                 # Checkpoint saving
    #                 if len(self.results_list) >= self.config.checkpoint_interval:
    #                     self._save_checkpoint(save_path, processed_count)

    #     # Final save
    #     self._save_final_results(save_path, resume_from is not None)

    #     # Final stats
    #     elapsed_time = time.time() - start_time
    #     self._log_final_statistics(total_voxels, processed_count, initial_processed_count, 
    #                              zero_mwf_count, below_threshold_count, processing_errors, 
    #                              elapsed_time, pre_stats)


    def run_analysis(self, save_dir: Optional[str] = None):
        """Run the complete analysis."""
        self.logger.info("Starting brain data analysis...")
        
        # Setup
        self.load_data()
        self.setup_parameters()
        self.setup_uniform_noise()
        
        # Analyze data before processing
        pre_stats = self.analyze_data_before_processing()
        
        # Create save directory
        if save_dir is None:
            date_now = date.today()
            day = date_now.strftime('%d')
            month = date_now.strftime('%B')[:3]
            year = date_now.strftime('%y')
            save_dir = f"data/Brain/results_{day}{month}{year}"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Process all voxels
        p, q, _ = self.brain_data.shape
        total_voxels = p * q
        processed_count = 0  # Valid results count
        skipped_count = 0    # None results count
        zero_mwf_count = 0   # Voxels where all MWF values are 0 or close to 0
        processing_errors = 0 # Processing errors count
        below_threshold_count = 0 # Below threshold count
        
        start_time = time.time()
        
        # Log initial statistics
        self.logger.info(f"Total voxels to process: {total_voxels}")
        self.logger.info(f"Brain data shape: {self.brain_data.shape}")
        self.logger.info(f"Expected valid voxels (from pre-analysis): {pre_stats['above_threshold']}")
        
        with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
            for x in range(p):
                for y in range(q):
                    # Check threshold first for accurate counting
                    if self.brain_data[x, y, 0] < self.config.data_threshold:
                        skipped_count += 1
                        below_threshold_count += 1
                        result = None
                    else:
                        try:
                            result = self.process_voxel(x, y)
                            if result is not None:
                                # Check if all MWF values are zero or close to zero
                                if self.check_zero_mwf(result):
                                    zero_mwf_count += 1
                                
                                self.results_list.append(result)
                                processed_count += 1
                            else:
                                skipped_count += 1
                                processing_errors += 1
                        except Exception as e:
                            skipped_count += 1
                            processing_errors += 1
                            self.logger.error(f"Unexpected error processing voxel ({x}, {y}): {e}")
                    
                    pbar.update(1)
                    
                    # Update progress bar description with counts
                    pbar.set_description(f"Valid: {processed_count}, Zero-MWF: {zero_mwf_count}, Skipped: {skipped_count}")
                    
                    # Log periodic statistics
                    current_total = processed_count + skipped_count
                    if current_total > 0 and current_total % 1000 == 0:  # Every 1000 voxels
                        valid_percentage = (processed_count / current_total) * 100
                        zero_mwf_percentage = (zero_mwf_count / processed_count) * 100 if processed_count > 0 else 0
                        self.logger.info(f"Progress: {current_total}/{total_voxels} voxels processed.")
                        self.logger.info(f"  Valid results: {processed_count} ({valid_percentage:.1f}%)")
                        self.logger.info(f"  Zero/Low MWF: {zero_mwf_count} ({zero_mwf_percentage:.1f}% of valid)")
                        self.logger.info(f"  Below threshold: {below_threshold_count}")
                        self.logger.info(f"  Processing errors: {processing_errors}")
                        self.logger.info(f"  Total skipped: {skipped_count}")
                    
                        # Final save and logging
                    if len(self.results_list) >= self.config.checkpoint_interval:
                        self._save_checkpoint(save_path, processed_count)

        self._save_final_results(save_path)

        elapsed_time = time.time() - start_time
        valid_pct = (processed_count / total_voxels) * 100 if total_voxels else 0
        zero_pct = (zero_mwf_count / processed_count) * 100 if processed_count else 0
        below_pct = (below_threshold_count / total_voxels) * 100 if total_voxels else 0
        error_pct = (processing_errors / total_voxels) * 100 if total_voxels else 0

        self.logger.info("=" * 70)
        self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
        self.logger.info("=" * 70)
        self.logger.info(f"Total voxels in dataset: {total_voxels}")
        self.logger.info(f"Processing time: {elapsed_time / 60:.2f} minutes")
        self.logger.info(f"Average time per voxel: {elapsed_time / total_voxels:.4f} seconds")
        self.logger.info("")
        self.logger.info(f"  Valid: {processed_count} ({valid_pct:.2f}%)")
        self.logger.info(f"  Skipped: {skipped_count} ({100 - valid_pct:.2f}%)")
        self.logger.info("")
        self.logger.info(f"  Below threshold: {below_threshold_count} ({below_pct:.2f}%)")
        self.logger.info(f"  Errors: {processing_errors} ({error_pct:.2f}%)")
        self.logger.info("")
        self.logger.info(f"  Zero MWF: {zero_mwf_count} ({zero_pct:.2f}% of valid)")
        self.logger.info(f"  Meaningful MWF: {processed_count - zero_mwf_count} ({100 - zero_pct:.2f}% of valid)")
        self.logger.info("")
        if processed_count:
            self.logger.info(f"  Avg time per valid result: {elapsed_time / processed_count:.4f} seconds")
            meaningful = processed_count - zero_mwf_count
            if meaningful:
                self.logger.info(f"  Avg time per meaningful result: {elapsed_time / meaningful:.4f} seconds")
        usable_pct = ((processed_count - zero_mwf_count) / total_voxels) * 100 if total_voxels else 0
        self.logger.info(f"  Usable data rate: {usable_pct:.2f}% of total voxels")
        if 'above_threshold' in self.pre_stats and self.pre_stats['above_threshold']:
            success_rate = (processed_count / self.pre_stats['above_threshold']) * 100
            self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
        if processed_count > 0:
            self.logger.info("")
            self.logger.info("MWF STATISTICS BY METHOD:")
            self._log_mwf_statistics_by_method()
            self.logger.info("="*70)

                    
        # # Periodic saving
        # if len(self.results_list) >= self.config.checkpoint_interval:
        #     self._save_checkpoint(save_path, processed_count)

        # # # Final save
        # # self._save_final_results(save_path)
        
        # # Final statistics
        # elapsed_time = time.time() - start_time
        valid_percentage = (processed_count / total_voxels) * 100 if total_voxels > 0 else 0
        zero_mwf_percentage = (zero_mwf_count / processed_count) * 100 if processed_count > 0 else 0
        below_threshold_percentage = (below_threshold_count / total_voxels) * 100 if total_voxels > 0 else 0
        processing_error_percentage = (processing_errors / total_voxels) * 100 if total_voxels > 0 else 0
        
        self.logger.info("="*70)
        self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
        self.logger.info("="*70)
        self.logger.info(f"Total voxels in dataset: {total_voxels}")
        self.logger.info(f"Processing time: {elapsed_time/60:.2f} minutes")
        self.logger.info(f"Average time per voxel: {elapsed_time/total_voxels:.4f} seconds")
        self.logger.info("")
        self.logger.info("PROCESSING RESULTS:")
        self.logger.info(f"  Valid results (not None): {processed_count} ({valid_percentage:.2f}%)")
        self.logger.info(f"  Total skipped results: {skipped_count} ({100-valid_percentage:.2f}%)")
        self.logger.info("")
        self.logger.info("SKIP REASONS BREAKDOWN:")
        self.logger.info(f"  Below data threshold ({self.config.data_threshold}): {below_threshold_count} ({below_threshold_percentage:.2f}%)")
        self.logger.info(f"  Processing errors: {processing_errors} ({processing_error_percentage:.2f}%)")
        self.logger.info("")
        self.logger.info("MWF ANALYSIS (for valid results):")
        self.logger.info(f"  Voxels with all MWF ≈ 0: {zero_mwf_count} ({zero_mwf_percentage:.2f}% of valid)")
        self.logger.info(f"  Voxels with meaningful MWF: {processed_count - zero_mwf_count} ({100-zero_mwf_percentage:.2f}% of valid)")
        self.logger.info("")
        self.logger.info("PERFORMANCE METRICS:")
        if processed_count > 0:
            self.logger.info(f"  Average time per valid result: {elapsed_time/processed_count:.4f} seconds")
            meaningful_results = processed_count - zero_mwf_count
            if meaningful_results > 0:
                self.logger.info(f"  Average time per meaningful result: {elapsed_time/meaningful_results:.4f} seconds")
        self.logger.info("")
        self.logger.info("DATA QUALITY ASSESSMENT:")
        usable_percentage = ((processed_count - zero_mwf_count) / total_voxels) * 100 if total_voxels > 0 else 0
        self.logger.info(f"  Overall usable data rate: {usable_percentage:.2f}% of total voxels")
        if pre_stats['above_threshold'] > 0:
            success_rate = (processed_count / pre_stats['above_threshold']) * 100
            self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
        self.logger.info("="*70)
        
        # Additional detailed breakdown by method (if you want to track MWF per method)

    def _load_checkpoint_state(self, resume_from: str, save_path: Path) -> tuple:
        """Load existing checkpoint data and return processed voxel coordinates."""
        resume_path = Path(resume_from)
        processed_voxels = set()
        total_processed = 0
        
        if resume_path.is_file():
            # Single checkpoint file
            self.logger.info(f"Loading checkpoint from: {resume_path}")
            df = pd.read_pickle(resume_path)
            processed_voxels = set(zip(df['X_val'], df['Y_val']))
            total_processed = len(df)
            
        elif resume_path.is_dir():
            # Directory with multiple checkpoint files
            checkpoint_files = list(resume_path.glob("checkpoint_*.pkl"))
            self.logger.info(f"Found {len(checkpoint_files)} checkpoint files in {resume_path}")
            
            all_dfs = []
            for file in sorted(checkpoint_files):
                self.logger.info(f"Loading {file.name}")
                df = pd.read_pickle(file)
                all_dfs.append(df)
            
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                processed_voxels = set(zip(combined_df['X_val'], combined_df['Y_val']))
                total_processed = len(combined_df)
                
                # Save combined checkpoint to current save directory
                combined_path = save_path / f"resume_checkpoint_{total_processed}.pkl"
                combined_df.to_pickle(combined_path)
                self.logger.info(f"Saved combined checkpoint: {combined_path}")
        
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {resume_from}")
        
        return processed_voxels, total_processed

    def _save_checkpoint(self, save_path: Path, count: int):
        """Save intermediate results."""
        if self.results_list:
            df = pd.concat(self.results_list, ignore_index=True)
            timestamp = int(time.time())
            checkpoint_file = save_path / f"checkpoint_{count}_{timestamp}.pkl"
            df.to_pickle(checkpoint_file)
            self.logger.info(f"Saved checkpoint with {len(df)} results to {checkpoint_file}")
            self.results_list.clear()
    
    def _save_final_results(self, save_path: Path, is_resume: bool = False):
        """Save final results, combining with existing checkpoints if resuming."""
        # Collect all checkpoint files (including any from resume)
        checkpoint_files = list(save_path.glob("checkpoint_*.pkl"))
        checkpoint_files.extend(list(save_path.glob("resume_checkpoint_*.pkl")))
        all_dfs = []
        
        # Load checkpoint files
        for file in sorted(checkpoint_files):
            self.logger.info(f"Loading final data from {file.name}")
            df = pd.read_pickle(file)
            all_dfs.append(df)
            file.unlink()  # Remove checkpoint file
        
        # Add remaining results
        if self.results_list:
            df = pd.concat(self.results_list, ignore_index=True)
            all_dfs.append(df)
        
        # Combine and save
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            # Remove duplicates (in case of overlapping checkpoints)
            original_len = len(final_df)
            final_df = final_df.drop_duplicates(subset=['X_val', 'Y_val'], keep='last')
            if len(final_df) < original_len:
                self.logger.info(f"Removed {original_len - len(final_df)} duplicate entries")
            
            # Generate filename
            date_now = date.today()
            day = date_now.strftime('%d')
            month = date_now.strftime('%B')[:3]
            year = date_now.strftime('%y')
            
            p, q, _ = self.brain_data.shape
            resume_suffix = "_resumed" if is_resume else ""
            filename = f"est_table_xcoordlen_{p}_ycoordlen_{q}_optimized_{day}{month}{year}{resume_suffix}.pkl"
            
            final_path = save_path / filename
            final_df.to_pickle(final_path)
            
            self.logger.info(f"Final results saved to {final_path}")
            self.logger.info(f"Total results: {len(final_df)}")

    def _log_final_statistics(self, total_voxels, processed_count, initial_processed_count, 
                            zero_mwf_count, below_threshold_count, processing_errors, 
                            elapsed_time, pre_stats):
        """Log comprehensive final statistics."""
        new_processed = processed_count - initial_processed_count
        valid_pct = (processed_count / total_voxels) * 100 if total_voxels else 0
        zero_pct = (zero_mwf_count / new_processed) * 100 if new_processed else 0
        below_pct = (below_threshold_count / total_voxels) * 100 if total_voxels else 0
        error_pct = (processing_errors / total_voxels) * 100 if total_voxels else 0
        meaningful = processed_count - zero_mwf_count
        usable_pct = (meaningful / total_voxels) * 100 if total_voxels else 0

        self.logger.info("=" * 70)
        self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
        self.logger.info("=" * 70)
        self.logger.info(f"Total voxels in dataset: {total_voxels}")
        self.logger.info(f"Previously processed: {initial_processed_count}")
        self.logger.info(f"Newly processed this session: {new_processed}")
        self.logger.info(f"Total processed: {processed_count}")
        self.logger.info(f"Processing time (this session): {elapsed_time / 60:.2f} minutes")
        if new_processed > 0:
            self.logger.info(f"Average time per new voxel: {elapsed_time / new_processed:.4f} seconds")
        self.logger.info("")
        self.logger.info("OVERALL PROCESSING RESULTS:")
        self.logger.info(f"  Total valid results: {processed_count} ({valid_pct:.2f}%)")
        self.logger.info(f"  Usable data rate: {usable_pct:.2f}% of total voxels")
        self.logger.info("=" * 70)


def main():
    """Main execution function with checkpoint resume option."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain Data Processor with Checkpoint Resume')
    parser.add_argument('--resume-from', type=str, help='Path to checkpoint file or directory to resume from')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    
    # Create processor and run analysis
    processor = BrainDataProcessor(config)
    processor.run_analysis(save_dir=args.save_dir, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
#     def run_analysis(self, save_dir: Optional[str] = None):
#         """Run the complete analysis."""
#         self.logger.info("Starting brain data analysis...")

#         # Setup
#         self.load_data()
#         self.setup_parameters()
#         self.setup_uniform_noise()
#         pre_stats = self.analyze_data_before_processing()

#         # Create save path
#         if save_dir is None:
#             date_now = date.today()
#             save_dir = f"data/Brain/results_{date_now.strftime('%d%b%y')}"
#         save_path = Path(save_dir)
#         save_path.mkdir(parents=True, exist_ok=True)

#         p, q, _ = self.brain_data.shape
#         total_voxels = p * q
#         voxel_coords = [(self, x, y) for x in range(p) for y in range(q)]

#         # Counters
#         processed_count = 0
#         zero_mwf_count = 0
#         skipped_count = 0
#         below_threshold_count = 0
#         processing_errors = 0
#         self.results_list = []

#         self.logger.info(f"Total voxels to process: {total_voxels}")
#         self.logger.info(f"Brain data shape: {self.brain_data.shape}")
#         self.logger.info(f"Expected valid voxels (from pre-analysis): {pre_stats['above_threshold']}")
#         start_time = time.time()
#         # with mp.Pool(processes=mp.cpu_count()) as pool:
#         #     with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
#         #         for status, x, y, result in pool.imap_unordered(self._voxel_worker, voxel_coords):
#         args_list = voxel_coords
#         with mp.Pool(processes=mp.cpu_count()) as pool:
#             with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
#                 for status, x, y, result in pool.imap_unordered(voxel_worker_wrapper, args_list):
#                     if status == 'valid':
#                         self.results_list.append(result)
#                         processed_count += 1
#                     elif status == 'zero_mwf':
#                         self.results_list.append(result)
#                         processed_count += 1
#                         zero_mwf_count += 1
#                     elif status == 'below_threshold':
#                         skipped_count += 1
#                         below_threshold_count += 1
#                     elif status == 'error':
#                         skipped_count += 1
#                         processing_errors += 1

#                     pbar.update(1)

#                     current_total = processed_count + skipped_count
#                     if current_total > 0 and current_total % 1000 == 0:
#                         valid_pct = (processed_count / current_total) * 100
#                         zero_pct = (zero_mwf_count / processed_count) * 100 if processed_count else 0
#                         self.logger.info(f"Progress: {current_total}/{total_voxels}")
#                         self.logger.info(f"  Valid: {processed_count} ({valid_pct:.1f}%)")
#                         self.logger.info(f"  Zero MWF: {zero_mwf_count} ({zero_pct:.1f}% of valid)")
#                         self.logger.info(f"  Below threshold: {below_threshold_count}")
#                         self.logger.info(f"  Errors: {processing_errors}")
                    
#                     if len(self.results_list) >= self.config.checkpoint_interval:
#                         self._save_checkpoint(save_path, processed_count)

#         # Final save
#         self._save_final_results(save_path)

#         # Final stats
#         elapsed_time = time.time() - start_time
#         valid_pct = (processed_count / total_voxels) * 100 if total_voxels else 0
#         zero_pct = (zero_mwf_count / processed_count) * 100 if processed_count else 0
#         below_pct = (below_threshold_count / total_voxels) * 100 if total_voxels else 0
#         error_pct = (processing_errors / total_voxels) * 100 if total_voxels else 0
#         meaningful = processed_count - zero_mwf_count
#         usable_pct = (meaningful / total_voxels) * 100 if total_voxels else 0
#         success_rate = (processed_count / pre_stats['above_threshold']) * 100 if pre_stats['above_threshold'] else 0

#         self.logger.info("=" * 70)
#         self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
#         self.logger.info("=" * 70)
#         self.logger.info(f"Total voxels in dataset: {total_voxels}")
#         self.logger.info(f"Processing time: {elapsed_time / 60:.2f} minutes")
#         self.logger.info(f"Average time per voxel: {elapsed_time / total_voxels:.4f} seconds")
#         self.logger.info("")
#         self.logger.info("PROCESSING RESULTS:")
#         self.logger.info(f"  Valid results: {processed_count} ({valid_pct:.2f}%)")
#         self.logger.info(f"  Skipped results: {skipped_count} ({100 - valid_pct:.2f}%)")
#         self.logger.info("")
#         self.logger.info("SKIP REASONS BREAKDOWN:")
#         self.logger.info(f"  Below threshold: {below_threshold_count} ({below_pct:.2f}%)")
#         self.logger.info(f"  Processing errors: {processing_errors} ({error_pct:.2f}%)")
#         self.logger.info("")
#         self.logger.info("MWF ANALYSIS (valid results):")
#         self.logger.info(f"  Zero MWF: {zero_mwf_count} ({zero_pct:.2f}%)")
#         self.logger.info(f"  Meaningful MWF: {meaningful} ({100 - zero_pct:.2f}%)")
#         self.logger.info("")
#         self.logger.info("PERFORMANCE METRICS:")
#         if processed_count:
#             self.logger.info(f"  Avg time per valid result: {elapsed_time / processed_count:.4f} seconds")
#         if meaningful:
#             self.logger.info(f"  Avg time per meaningful result: {elapsed_time / meaningful:.4f} seconds")
#         self.logger.info("")
#         self.logger.info("DATA QUALITY ASSESSMENT:")
#         self.logger.info(f"  Usable data rate: {usable_pct:.2f}% of total voxels")
#         self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
#         self.logger.info("=" * 70)

#         if processed_count:
#             self.logger.info("MWF STATISTICS BY METHOD:")
#             self._log_mwf_statistics_by_method()
#             self.logger.info("=" * 70)

#     def _save_checkpoint(self, save_path: Path, count: int):
#         """Save intermediate results."""
#         if self.results_list:
#             df = pd.concat(self.results_list, ignore_index=True)
#             checkpoint_file = save_path / f"checkpoint_{count}.pkl"
#             df.to_pickle(checkpoint_file)
#             self.logger.info(f"Saved checkpoint with {len(df)} results")
#             self.results_list.clear()
    
#     def _save_final_results(self, save_path: Path):
#         """Save final results."""
#         # Collect all checkpoint files
#         checkpoint_files = list(save_path.glob("checkpoint_*.pkl"))
#         all_dfs = []
        
#         # Load checkpoint files
#         for file in checkpoint_files:
#             df = pd.read_pickle(file)
#             all_dfs.append(df)
#             file.unlink()  # Remove checkpoint file
        
#         # Add remaining results
#         if self.results_list:
#             df = pd.concat(self.results_list, ignore_index=True)
#             all_dfs.append(df)
        
#         # Combine and save
#         if all_dfs:
#             final_df = pd.concat(all_dfs, ignore_index=True)
            
#             # Generate filename
#             date_now = date.today()
#             day = date_now.strftime('%d')
#             month = date_now.strftime('%B')[:3]
#             year = date_now.strftime('%y')
            
#             p, q, _ = self.brain_data.shape
#             filename = f"est_table_xcoordlen_{p}_ycoordlen_{q}_optimized_{day}{month}{year}.pkl"
            
#             final_path = save_path / filename
#             final_df.to_pickle(final_path)
            
#             self.logger.info(f"Final results saved to {final_path}")
#             self.logger.info(f"Total results: {len(final_df)}")


# def main():
#     """Main execution function."""
#     # Create configuration
#     config = Config()
    
#     # Override any config parameters as needed
#     # config.uniform_noise = False  # Example override
    
#     # Create processor and run analysis
#     processor = BrainDataProcessor(config)
#     processor.run_analysis()


# if __name__ == "__main__":
#     main()


    # def run_analysis(self, save_dir: Optional[str] = None):
    #     """Run the complete analysis."""
    #     self.logger.info("Starting brain data analysis...")
        
    #     # Setup
    #     self.load_data()
    #     self.setup_parameters()
    #     self.setup_uniform_noise()
        
    #     # Analyze data before processing
    #     pre_stats = self.analyze_data_before_processing()
        
    #     # Create save directory
    #     if save_dir is None:
    #         date_now = date.today()
    #         day = date_now.strftime('%d')
    #         month = date_now.strftime('%B')[:3]
    #         year = date_now.strftime('%y')
    #         save_dir = f"data/Brain/results_{day}{month}{year}"
        
    #     save_path = Path(save_dir)
    #     save_path.mkdir(parents=True, exist_ok=True)
        
    #     # Process all voxels
    #     p, q, _ = self.brain_data.shape
    #     total_voxels = p * q
    #     processed_count = 0  # Valid results count
    #     skipped_count = 0    # None results count
    #     zero_mwf_count = 0   # Voxels where all MWF values are 0 or close to 0
    #     processing_errors = 0 # Processing errors count
    #     below_threshold_count = 0 # Below threshold count
        
    #     start_time = time.time()
        
    #     # Log initial statistics
    #     self.logger.info(f"Total voxels to process: {total_voxels}")
    #     self.logger.info(f"Brain data shape: {self.brain_data.shape}")
    #     self.logger.info(f"Expected valid voxels (from pre-analysis): {pre_stats['above_threshold']}")
        
    #     with tqdm(total=total_voxels, desc="Processing voxels") as pbar:
    #         for x in range(p):
    #             for y in range(q):
    #                 # Check threshold first for accurate counting
    #                 if self.brain_data[x, y, 0] < self.config.data_threshold:
    #                     skipped_count += 1
    #                     below_threshold_count += 1
    #                     result = None
    #                 else:
    #                     try:
    #                         result = self.process_voxel(x, y)
    #                         if result is not None:
    #                             # Check if all MWF values are zero or close to zero
    #                             if self.check_zero_mwf(result):
    #                                 zero_mwf_count += 1
                                
    #                             self.results_list.append(result)
    #                             processed_count += 1
    #                         else:
    #                             skipped_count += 1
    #                             processing_errors += 1
    #                     except Exception as e:
    #                         skipped_count += 1
    #                         processing_errors += 1
    #                         self.logger.error(f"Unexpected error processing voxel ({x}, {y}): {e}")
                    
    #                 pbar.update(1)
                    
    #                 # Update progress bar description with counts
    #                 pbar.set_description(f"Valid: {processed_count}, Zero-MWF: {zero_mwf_count}, Skipped: {skipped_count}")
                    
    #                 # Log periodic statistics
    #                 current_total = processed_count + skipped_count
    #                 if current_total > 0 and current_total % 1000 == 0:  # Every 1000 voxels
    #                     valid_percentage = (processed_count / current_total) * 100
    #                     zero_mwf_percentage = (zero_mwf_count / processed_count) * 100 if processed_count > 0 else 0
    #                     self.logger.info(f"Progress: {current_total}/{total_voxels} voxels processed.")
    #                     self.logger.info(f"  Valid results: {processed_count} ({valid_percentage:.1f}%)")
    #                     self.logger.info(f"  Zero/Low MWF: {zero_mwf_count} ({zero_mwf_percentage:.1f}% of valid)")
    #                     self.logger.info(f"  Below threshold: {below_threshold_count}")
    #                     self.logger.info(f"  Processing errors: {processing_errors}")
    #                     self.logger.info(f"  Total skipped: {skipped_count}")
                    
    #                     # Final save and logging
    #                 if len(self.results_list) >= self.config.checkpoint_interval:
    #                     self._save_checkpoint(save_path, processed_count)

    #     self._save_final_results(save_path)

    #     elapsed_time = time.time() - start_time
    #     valid_pct = (processed_count / total_voxels) * 100 if total_voxels else 0
    #     zero_pct = (zero_mwf_count / processed_count) * 100 if processed_count else 0
    #     below_pct = (below_threshold_count / total_voxels) * 100 if total_voxels else 0
    #     error_pct = (processing_errors / total_voxels) * 100 if total_voxels else 0

    #     self.logger.info("=" * 70)
    #     self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
    #     self.logger.info("=" * 70)
    #     self.logger.info(f"Total voxels in dataset: {total_voxels}")
    #     self.logger.info(f"Processing time: {elapsed_time / 60:.2f} minutes")
    #     self.logger.info(f"Average time per voxel: {elapsed_time / total_voxels:.4f} seconds")
    #     self.logger.info("")
    #     self.logger.info(f"  Valid: {processed_count} ({valid_pct:.2f}%)")
    #     self.logger.info(f"  Skipped: {skipped_count} ({100 - valid_pct:.2f}%)")
    #     self.logger.info("")
    #     self.logger.info(f"  Below threshold: {below_threshold_count} ({below_pct:.2f}%)")
    #     self.logger.info(f"  Errors: {processing_errors} ({error_pct:.2f}%)")
    #     self.logger.info("")
    #     self.logger.info(f"  Zero MWF: {zero_mwf_count} ({zero_pct:.2f}% of valid)")
    #     self.logger.info(f"  Meaningful MWF: {processed_count - zero_mwf_count} ({100 - zero_pct:.2f}% of valid)")
    #     self.logger.info("")
    #     if processed_count:
    #         self.logger.info(f"  Avg time per valid result: {elapsed_time / processed_count:.4f} seconds")
    #         meaningful = processed_count - zero_mwf_count
    #         if meaningful:
    #             self.logger.info(f"  Avg time per meaningful result: {elapsed_time / meaningful:.4f} seconds")
    #     usable_pct = ((processed_count - zero_mwf_count) / total_voxels) * 100 if total_voxels else 0
    #     self.logger.info(f"  Usable data rate: {usable_pct:.2f}% of total voxels")
    #     if 'above_threshold' in self.pre_stats and self.pre_stats['above_threshold']:
    #         success_rate = (processed_count / self.pre_stats['above_threshold']) * 100
    #         self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
    #     if processed_count > 0:
    #         self.logger.info("")
    #         self.logger.info("MWF STATISTICS BY METHOD:")
    #         self._log_mwf_statistics_by_method()
    #         self.logger.info("="*70)

                    
    #     # # Periodic saving
    #     # if len(self.results_list) >= self.config.checkpoint_interval:
    #     #     self._save_checkpoint(save_path, processed_count)

    #     # # # Final save
    #     # # self._save_final_results(save_path)
        
    #     # # Final statistics
    #     # elapsed_time = time.time() - start_time
    #     valid_percentage = (processed_count / total_voxels) * 100 if total_voxels > 0 else 0
    #     zero_mwf_percentage = (zero_mwf_count / processed_count) * 100 if processed_count > 0 else 0
    #     below_threshold_percentage = (below_threshold_count / total_voxels) * 100 if total_voxels > 0 else 0
    #     processing_error_percentage = (processing_errors / total_voxels) * 100 if total_voxels > 0 else 0
        
    #     self.logger.info("="*70)
    #     self.logger.info("ANALYSIS COMPLETE - FINAL COMPREHENSIVE STATISTICS")
    #     self.logger.info("="*70)
    #     self.logger.info(f"Total voxels in dataset: {total_voxels}")
    #     self.logger.info(f"Processing time: {elapsed_time/60:.2f} minutes")
    #     self.logger.info(f"Average time per voxel: {elapsed_time/total_voxels:.4f} seconds")
    #     self.logger.info("")
    #     self.logger.info("PROCESSING RESULTS:")
    #     self.logger.info(f"  Valid results (not None): {processed_count} ({valid_percentage:.2f}%)")
    #     self.logger.info(f"  Total skipped results: {skipped_count} ({100-valid_percentage:.2f}%)")
    #     self.logger.info("")
    #     self.logger.info("SKIP REASONS BREAKDOWN:")
    #     self.logger.info(f"  Below data threshold ({self.config.data_threshold}): {below_threshold_count} ({below_threshold_percentage:.2f}%)")
    #     self.logger.info(f"  Processing errors: {processing_errors} ({processing_error_percentage:.2f}%)")
    #     self.logger.info("")
    #     self.logger.info("MWF ANALYSIS (for valid results):")
    #     self.logger.info(f"  Voxels with all MWF ≈ 0: {zero_mwf_count} ({zero_mwf_percentage:.2f}% of valid)")
    #     self.logger.info(f"  Voxels with meaningful MWF: {processed_count - zero_mwf_count} ({100-zero_mwf_percentage:.2f}% of valid)")
    #     self.logger.info("")
    #     self.logger.info("PERFORMANCE METRICS:")
    #     if processed_count > 0:
    #         self.logger.info(f"  Average time per valid result: {elapsed_time/processed_count:.4f} seconds")
    #         meaningful_results = processed_count - zero_mwf_count
    #         if meaningful_results > 0:
    #             self.logger.info(f"  Average time per meaningful result: {elapsed_time/meaningful_results:.4f} seconds")
    #     self.logger.info("")
    #     self.logger.info("DATA QUALITY ASSESSMENT:")
    #     usable_percentage = ((processed_count - zero_mwf_count) / total_voxels) * 100 if total_voxels > 0 else 0
    #     self.logger.info(f"  Overall usable data rate: {usable_percentage:.2f}% of total voxels")
    #     if pre_stats['above_threshold'] > 0:
    #         success_rate = (processed_count / pre_stats['above_threshold']) * 100
    #         self.logger.info(f"  Processing success rate: {success_rate:.2f}% of above-threshold voxels")
    #     self.logger.info("="*70)
        
    #     # Additional detailed breakdown by method (if you want to track MWF per method)


# import sys, os, numpy as np, pickle, logging, time, random, scipy, timeit, unittest
# import matplotlib.pyplot as plt, seaborn as sns, matplotlib.ticker as ticker
# from scipy.stats import norm as normsci, wasserstein_distance
# from scipy.linalg import norm as linalg_norm, svd
# from scipy.optimize import nnls
# from scipy.integrate import simpson
# import pandas as pd
# import cvxpy as cp
# from regu.csvd import csvd
# from regu.discrep import discrep
# from regu.l_curve import l_curve
# from regu.nonnegtik_hnorm import nonnegtik_hnorm
# from Utilities_functions.discrep_L2 import *
# from Utilities_functions.GCV_NNLS import GCV_NNLS
# from Utilities_functions.Lcurve import Lcurve
# from Utilities_functions.tikhonov_vec import tikhonov_vec
# from Utilities_functions.pasha_gcv import Tikhonov
# from Simulations.LRalgo import *
# from Simulations.resolutionanalysis import find_min_between_peaks, check_resolution
# import multiprocess as mp
# from multiprocessing import Pool, freeze_support, set_start_method
# from tqdm import tqdm
# from io import StringIO
# import cProfile, pstats, functools
# from Simulations.upenzama import UPEN_Zama
# import matlab.engine
# from datetime import date
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
# preset_noise = False
# # presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25 copy.pkl"
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
# # presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_14Apr25\est_table_xcoordlen_313_ycoordlen_313_filtered_noise_addition_uniform_noise_UPEN_LR1D2D14Apr25.pkl"
# unif_noise = True

# print("Setting system path")
# sys.path.append(".")

# # Logging setup
# log_dir = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1"
# try:
#     os.makedirs(log_dir, exist_ok=True)
# except Exception as e:
#     print(f"Error creating log directory: {e}")
#     sys.exit(1)

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logger.handlers = []

# brain_log_path = os.path.join(log_dir, 'braindatascript.log')
# app_log_path = os.path.join(log_dir, 'app.log')
# file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# try:
#     file_handler = logging.FileHandler(brain_log_path)
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(file_format)
#     logger.addHandler(file_handler)
# except Exception as e:
#     print(f"Error setting up braindatascript.log: {e}")

# try:
#     app_handler = logging.FileHandler(app_log_path)
#     app_handler.setLevel(logging.DEBUG)
#     app_handler.setFormatter(file_format)
#     logger.addHandler(app_handler)
# except Exception as e:
#     print(f"Error setting up app.log: {e}")

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# console_handler.setFormatter(file_format)
# logger.addHandler(console_handler)

# logging.info("Logging initialized successfully")

# print("setting license path")
# logging.info("setting license path")
# mosek_license_path = r"/home/kimjosy/LocReg_Regularization-1/mosek/mosek.lic"
# os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# logging.info(f'MOSEK License Set from {mosek_license_path}')

# parent = os.path.dirname(os.path.abspath(''))
# sys.path.append(parent)
# cwd_temp = os.getcwd()
# base_file = 'LocReg_Regularization-1'
# cwd_cut = f'{cwd_temp.split(base_file, 1)[0]}{base_file}/'

# pat_tag = "MRR"
# series_tag = "BrainData_Images"
# simulation_save_folder = f"SimulationSets/{pat_tag}/{series_tag}"
# cwd_full = cwd_cut + simulation_save_folder
# logging.info(f"Save Folder for Brain Images {cwd_full})")

# addingNoise = True

# brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]
# SNR_map = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\SNRmap\new_SNR_Map.mat")["SNR_MAP"]

# logging.info(f"brain_data shape {brain_data.shape}")
# logging.info(f"SNR_map shape {SNR_map.shape}")
# logging.info("brain_data and SNR_map from Chuan have been successfully loaded")
# print("SNR_map.shape", SNR_map.shape)
# # Iterator for Parallel Processing of Brain Data
# p, q, s = brain_data.shape

# # Ensure minimum SNR of 1 (avoid division by zero or log issues)
# SNR_map = np.where(SNR_map == 0, 1, SNR_map)
# print("Max SNR_map:", np.max(SNR_map))
# print("Min SNR_map:", np.min(SNR_map))

# # Regularization Span Level
# SpanReg_level = 800

# # Iterator over all (x, y) pixel locations
# target_iterator = [(a, b) for a in range(p) for b in range(q)]
# print("Target Iterator:", target_iterator)
# logging.debug(f'Target Iterator Length: {len(target_iterator)}')

# # Naming for Saving Output
# date_now = date.today()
# day = date_now.strftime('%d')
# month = date_now.strftime('%B')[:3]
# year = date_now.strftime('%y')
# data_path = f"data/Brain/results_{day}{month}{year}"
# add_tag = f"xcoordlen_{p}_ycoordlen_{q}_NESMA_filtered_NA_GCV_LR012_UPEN"
# data_tag = f"est_table_{add_tag}{day}{month}{year}"
# data_folder = os.path.join(os.getcwd(), data_path)
# os.makedirs(data_folder, exist_ok=True)
# logging.info(f"Save Folder for Final Estimates Table: {data_folder}")

# # Parallelization Settings
# parallel = False
# num_cpus_avail = os.cpu_count()

# # LocReg Hyperparameters
# eps1 = 1e-2
# ep_min = 1e-2
# eps_cut = 1.2
# eps_floor = 1e-4
# exp = 0.5
# feedback = True
# lam_ini_val = "LCurve"
# gamma_init = 0.5
# maxiter = 500

# # CVXPY Global Parameters
# eps = 1e-2

# # Error Metric Choice
# err_type = "Wass. Score"

# # Lambda (Regularization) Space
# Lambda = np.logspace(-6, 1, 50).reshape(-1, 1)

# # --- Utility Functions ---

# def wass_error(IdealModel, reconstr):
#     return wasserstein_distance(IdealModel, reconstr)

# def l2_error(IdealModel, reconstr):
#     return linalg_norm(IdealModel - reconstr) / linalg_norm(IdealModel)

# def cvxpy_tikhreg(Lambda, G, data_noisy):
#     """
#     Placeholder for an alternative CVXPY-based Tikhonov solver.
#     Currently bypassed in favor of nonnegtik_hnorm.
#     """
#     return nonnegtik_hnorm

# def add_noise(data, SNR):
#     """
#     Add Gaussian noise to data according to specified SNR.
#     """
#     SD_noise = np.max(np.abs(data)) / SNR
#     noise = np.random.normal(0, SD_noise, size=data.shape)
#     dat_noisy = data + noise
#     return dat_noisy, noise

# def choose_error(pdf1, pdf2, err_type):
#     """
#     Choose error type ("Wass. Score" or L2).
#     """
#     if err_type in ["Wass. Score", "Wassterstein"]:
#         return wass_error(pdf1, pdf2)
#     else:
#         return l2_error(pdf1, pdf2)

# def minimize_OP(Lambda, data_noisy, G, nT2, g):
#     """
#     Oracle solution: selects optimal lambda minimizing error vs ground truth.
#     """
#     OP_x_lc_vec = np.zeros((nT2, len(Lambda)))
#     OP_rhos = np.zeros(len(Lambda))

#     for j in range(len(Lambda)):
#         sol = nonnegtik_hnorm(G, data_noisy, Lambda[j], '0', nargin=4)[0]
#         OP_x_lc_vec[:, j] = sol
#         OP_rhos[j] = choose_error(g, sol, err_type="Wass. Score")

#     min_index = np.argmin(OP_rhos)
#     f_rec_OP_grid = OP_x_lc_vec[:, min_index]
#     OP_min_alpha1 = Lambda[min_index][0]
#     min_rhos = OP_rhos[min_index]

#     return f_rec_OP_grid, OP_min_alpha1, min_rhos, min_index

# def compute_MWF(f_rec, T2, Myelin_idx):
#     """
#     Compute Myelin Water Fraction (MWF) from the reconstruction.
#     """
#     try:
#         f_rec_normalized = f_rec / (np.sum(f_rec) * dT)
#     except ZeroDivisionError:
#         epsilon = 1e-4
#         print("Division by zero encountered, using epsilon:", epsilon)
#         f_rec_normalized = f_rec / epsilon

#     total_MWF = np.cumsum(f_rec_normalized)
#     MWF = total_MWF[Myelin_idx[-1][-1]]
#     return f_rec_normalized, MWF

# def curve_plot(method, x_coord, y_coord, frec, curr_data, lambda_vals, curr_SNR, MWF, filepath):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].plot(T2, frec)
#     axs[0].set(title='T2 vs f_rec', xlabel='T2', ylabel='f_rec')
#     axs[1].plot(TE, curr_data)
#     axs[1].set(title='TE vs Decay Data', xlabel='TE', ylabel='curr_data')
#     axs[2].plot(T2, [lambda_vals]*len(T2))
#     axs[2].set(title='T2 vs Lambda', xlabel='T2', ylabel='lambda')
#     fig.suptitle(f'{method} Plots for x={x_coord}, y={y_coord} | SNR={curr_SNR}, MWF={MWF}', fontsize=16)
#     plt.savefig(f"{filepath}/{method}_recon_xcoord{x_coord}_ycoord{y_coord}.png")
#     plt.close(fig)

# def calculate_noise(signals):
#     mean_sig = np.mean(signals, axis=0)
#     noise_stddev = np.sqrt(np.sum((signals - mean_sig)**2, axis=0))[0]
#     return mean_sig[0], noise_stddev

# def filter_and_compute_MWF(reconstr, tol=1e-6):
#     if np.all(np.abs(reconstr) < tol) or np.all(reconstr[:-1] == 0):
#         return reconstr, 0, 1
#     return compute_MWF(reconstr, T2, Myelin_idx) + (0,)

# def get_signals(coord_pairs, mask_array, unfiltered_arr, A, dT):
#     signals, SNRs = [], []
#     for x, y in coord_pairs:
#         signals.append(unfiltered_arr[x, y, :])
#         SNRs.append(SNR_map[x, y])
#         print(f"Coordinate: ({x}, {y}), Mask value: {mask_array[x, y]}")
#     return np.array(signals), np.array(SNRs)

# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)
# preset_noise = False
# presetfilepath = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_18Apr25\est_table_xcoordlen_313_ycoordlen_313_NESMA_filtered_myelinmaps18Apr25.pkl"
# unif_noise = True

# def generate_spanregbrain(i_param_combo, seed=None):
#     feature_df = pd.DataFrame(columns=[
#         "X_val", "Y_val", "curr_data", "noise", "curr_SNR",
#         "ref_estimate", "LR_estimate", "LR1D_estimate", "LR2D_estimate",
#         "GCV_estimate", "UPEN_estimate",
#         "MWF_Ref", "MWF_LR", "MWF_LR1D", "MWF_LR2D", "MWF_GCV", "MWF_UPEN",
#         "Lam_LR", "Lam_LR1D", "Lam_LR2D", "Lam_GCV", "Lam_UPEN", "Lam_Ref",
#         "GCV_Flag_Val", "LR_Flag_Val", "LR1D_Flag_Val", "LR2D_Flag_Val", "UPEN_Flag_Val", "Ref_Flag_Val"
#     ])
#     x_coord, y_coord = i_param_combo if not parallel else target_iterator[i_param_combo]
#     if brain_data[x_coord, y_coord][0] < 50:
#         print(f"not satisfies <50 requirement for {x_coord} and {y_coord}")
#         return feature_df

#     curr_data = brain_data[x_coord, y_coord, :]
#     curr_SNR = SNR_map[x_coord, y_coord]

#     sol1 = nonnegtik_hnorm(A, curr_data, 0, '0', nargin=4)[0]
#     factor = np.sum(sol1) * dT
#     curr_data = curr_data / factor

#     ref_rec, ref_lamb = GCV_NNLS(curr_data, A, Lambda)
#     ref_rec = ref_rec[:, 0]
#     ref_lamb = np.squeeze(ref_lamb)
#     ref_rec, ref_MWF, ref_Flag_Val = filter_and_compute_MWF(ref_rec)

#     data_gt = A @ ref_rec
#     curr_data = data_gt + unif_noise_val

#     noisy_f_rec_GCV, noisy_lambda_GCV = GCV_NNLS(curr_data, A, Lambda)
#     noisy_f_rec_GCV = noisy_f_rec_GCV[:, 0]
#     noisy_lambda_GCV = np.squeeze(noisy_lambda_GCV)
#     noisy_f_rec_GCV, noisy_MWF_GCV, GCV_Flag_Val = filter_and_compute_MWF(noisy_f_rec_GCV)

#     LRIto_ini_lam = noisy_lambda_GCV
#     noisy_f_rec_LR, noisy_lambda_LR, _, _, _ = LocReg_Ito_mod(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
#     noisy_f_rec_LR, noisy_MWF_LR, LR_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR)

#     noisy_f_rec_LR1D, noisy_lambda_LR1D, _, _, _ = LocReg_Ito_mod_deriv(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
#     noisy_f_rec_LR1D, noisy_MWF_LR1D, LR1D_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR1D)

#     noisy_f_rec_LR2D, noisy_lambda_LR2D, _, _, _ = LocReg_Ito_mod_deriv2(curr_data, A, LRIto_ini_lam, gamma_init, maxiter=50)
#     noisy_f_rec_LR2D, noisy_MWF_LR2D, LR2D_Flag_Val = filter_and_compute_MWF(noisy_f_rec_LR2D)

#     std_dev = np.std(curr_data[-5:])
#     SNR_est = np.max(np.abs(curr_data)) / std_dev
#     threshold = 1.05 * np.sqrt(A.shape[0]) * np.max(curr_data) / SNR_est
#     noise_norm = threshold

#     noisy_f_rec_UPEN, noisy_lambda_UPEN = UPEN_Zama(A, curr_data, noisy_f_rec_GCV, noise_norm, 1e-3, 50, 1e-5)
#     noisy_f_rec_UPEN, noisy_MWF_UPEN, UPEN_Flag_Val = filter_and_compute_MWF(noisy_f_rec_UPEN)

#     feature_df.loc[0] = [
#         x_coord, y_coord, curr_data, unif_noise_val, curr_SNR,
#         ref_rec, noisy_f_rec_LR, noisy_f_rec_LR1D, noisy_f_rec_LR2D,
#         noisy_f_rec_GCV, noisy_f_rec_UPEN,
#         ref_MWF, noisy_MWF_LR, noisy_MWF_LR1D, noisy_MWF_LR2D, noisy_MWF_GCV, noisy_MWF_UPEN,
#         noisy_lambda_LR, noisy_lambda_LR1D, noisy_lambda_LR2D, noisy_lambda_GCV, noisy_lambda_UPEN, ref_lamb,
#         GCV_Flag_Val, LR_Flag_Val, LR1D_Flag_Val, LR2D_Flag_Val, UPEN_Flag_Val, ref_Flag_Val
#     ]
#     print(f"completed dataframe for x {x_coord} and y {y_coord}")
#     return feature_df

# if __name__ == "__main__":
#     logging.info("Script started.")
#     freeze_support()
#     dTE = 11.3
#     n = 32
#     TE = dTE * np.linspace(1, n, n)
#     m = 150
#     T2 = np.linspace(10, 200, m)
#     A = np.zeros((n, m))
#     dT = T2[1] - T2[0]
#     logging.info(f"T2 range is from 10ms to 200ms with {m} discretizations")
#     logging.info(f"dT is {dT}")
#     logging.info(f"TE range is {TE}")

#     for i in range(n):
#         for j in range(m):
#             A[i, j] = np.exp(-TE[i] / T2[j]) * dT

#     unif_noise = True
#     if unif_noise:
#         num_signals = 1000
#         coord_pairs = set()
#         mask_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\brain\masks\new_mask.mat"
#         mask = scipy.io.loadmat(mask_path)["new_BW"]
#         brain_data = scipy.io.loadmat(r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\braindata\cleaned_brain_data (1).mat")["final_data_2"]

#         while len(coord_pairs) < num_signals:
#             x = random.randint(130, 200)
#             y = random.randint(100, 200)
#             if mask[x, y] == 0:
#                 coord_pairs.add((x, y))
#         coord_pairs = list(coord_pairs)

#         signals, SNRs = get_signals(coord_pairs, mask, brain_data, A, dT)
#         mean_sig = np.mean(signals, axis=0)
#         sol2 = nnls(A, mean_sig)[0]
#         factor2 = np.sum(sol2) * dT
#         mean_sig = mean_sig / factor2

#         tail_length = 3
#         tail = mean_sig[-tail_length:]
#         tail_std = np.std(tail)

#         unif_noise_val = np.random.normal(0, tail_std, size=32)
#         plt.figure()
#         plt.plot(unif_noise_val)
#         plt.xlabel('Time/Index')
#         plt.ylabel('Noise Standard Deviation')
#         plt.title('Noise Standard Deviation Across Signals')
#         plt.grid(True)
#         plt.savefig("testfignoise.png")
#         plt.close()

#         plt.figure()
#         plt.plot(mean_sig)
#         plt.xlabel('TE')
#         plt.ylabel('Amplitude')
#         plt.title('Mean Signal')
#         plt.grid(True)
#         plt.savefig("testfigmeansig.png")
#         plt.close()

#     logging.info(f"Kernel matrix is size {A.shape} and is form np.exp(-TE[i]/T2[j]) * dT")

#     LS_estimates = np.zeros((p, q, m))
#     MWF_LS = np.zeros((p, q))
#     LR_estimates = np.zeros((p, q, m))
#     MWF_LR = np.zeros((p, q))
#     LC_estimates = np.zeros((p, q, m))
#     MWF_LC = np.zeros((p, q))
#     GCV_estimates = np.zeros((p, q, m))
#     MWF_GCV = np.zeros((p, q))
#     DP_estimates = np.zeros((p, q, m))
#     MWF_DP = np.zeros((p, q))
#     OR_estimates = np.zeros((p, q, m))
#     MWF_OR = np.zeros((p, q))

#     Myelin_idx = np.where(T2 <= 40)
#     logging.info("We define myelin index to be less than 40 ms.")
#     logging.info("Since this is experimental patient brain data, we do not have the ground truth. Set the ground truth to be GCV.")

#     lis = []
#     checkpoint_file = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\data\Brain\results_16May25\checkpoint.pkl"
#     temp_checkpoint_prefix = f"{data_folder}/temp_checkpoint_"
#     checkpoint_interval = 1000
#     checkpoint_time_interval = 900
#     last_checkpoint_time = time.time()
#     temp_checkpoint_count = 0

#     if os.path.exists(checkpoint_file):
#         with open(checkpoint_file, 'rb') as f:
#             checkpoint_data = pickle.load(f)
#             lis = checkpoint_data['lis']
#             start_j = checkpoint_data['j']
#             start_k = checkpoint_data['k']
#             temp_checkpoint_count = checkpoint_data['temp_checkpoint_count']
#             logging.info(f"Resumed from checkpoint at j={start_j}, k={start_k}")
#     else:
#         start_j = 0
#         start_k = 0
#         logging.info("No checkpoint found, starting fresh.")

#     start_time = time.time()
#     total_iterations = p * q
#     completed_iterations = 0

#     try:
#         for j in tqdm(range(start_j, p), desc="Processing rows"):
#             for k in tqdm(range(start_k if j == start_j else 0, q), desc=f"Cols in row {j}", leave=False):
#                 estimates_dataframe = generate_spanregbrain((j, k))
#                 if not estimates_dataframe.empty:
#                     lis.append(estimates_dataframe)
#                     logging.info(f"Processed voxel ({j}, {k}), total processed: {len(lis)}")

#                 completed_iterations += 1
#                 elapsed_time = time.time() - start_time
#                 estimated_time_remaining = (elapsed_time / completed_iterations) * (total_iterations - completed_iterations)
#                 logging.info(f"Processed row {j}/{p}, column {k}/{q}, "
#                              f"elapsed time: {elapsed_time / 60:.2f} minutes, "
#                              f"estimated time left: {estimated_time_remaining / 60:.2f} minutes.")

#                 if len(lis) >= checkpoint_interval or (time.time() - last_checkpoint_time) >= checkpoint_time_interval:
#                     if lis:
#                         temp_df = pd.concat(lis, ignore_index=True)
#                         temp_checkpoint_file = f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl"
#                         temp_df.to_pickle(temp_checkpoint_file)
#                         logging.info(f"Saved temp checkpoint: {temp_checkpoint_file}")
#                         temp_checkpoint_count += 1
#                         lis = []

#                     checkpoint_data = {
#                         'lis': lis,
#                         'j': j,
#                         'k': k,
#                         'temp_checkpoint_count': temp_checkpoint_count
#                     }
#                     with open(checkpoint_file, 'wb') as f:
#                         pickle.dump(checkpoint_data, f)
#                     last_checkpoint_time = time.time()

#             start_k = 0

#         final_dfs = []
#         for i in range(temp_checkpoint_count):
#             temp_file = f"{temp_checkpoint_prefix}{i}.pkl"
#             if os.path.exists(temp_file):
#                 final_dfs.append(pd.read_pickle(temp_file))
#                 os.remove(temp_file)
#                 logging.info(f"Cleaned up temp checkpoint {temp_file}")
#         if lis:
#             final_dfs.append(pd.concat(lis, ignore_index=True))

#         if final_dfs:
#             df = pd.concat(final_dfs, ignore_index=True)
#             df.to_pickle(os.path.join(data_folder, f"{data_tag}.pkl"))
#             logging.info(f"Final results saved to {data_folder}/{data_tag}.pkl")

#         if os.path.exists(checkpoint_file):
#             os.remove(checkpoint_file)
#             logging.info("Removed main checkpoint after successful completion.")

#     except Exception as e:
#         logging.error(f"Error: {e}")
#         if lis:
#             temp_df = pd.concat(lis, ignore_index=True)
#             temp_df.to_pickle(f"{temp_checkpoint_prefix}{temp_checkpoint_count}.pkl")
#         with open(checkpoint_file, 'wb') as f:
#             pickle.dump({
#                 'lis': lis,
#                 'j': j,
#                 'k': k,
#                 'temp_checkpoint_count': temp_checkpoint_count
#             }, f)
#         raise
