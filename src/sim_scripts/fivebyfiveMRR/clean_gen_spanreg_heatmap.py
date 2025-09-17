from dataclasses import dataclass
from typing import Tuple, List, Optional
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    n_sim: int = 4
    SNR_value: int = 1000
    npeaks: int = 2
    nsigma: int = 5
    nrps: int = None  # Will be calculated
    Kmax: int = 500
    beta_0: float = 1e-7
    tol_lam: float = 1e-5
    gamma_init: float = 0.5
    reg_param_lb: float = -5
    reg_param_ub: float = 1
    N_reg: int = 25
    err_type: str = "WassScore"
    lam_ini_val: str = "GCV"
    dist_type: str = "narrowL_broadR_parallel"
    parallel: bool = False
    show: bool = True
    preset_noise: bool = False

class RegularizationSimulator:
    """Main class for running regularization method comparison simulations."""
    
    def __init__(self, config: SimulationConfig, file_path: str):
        self.config = config
        self.file_path = file_path
        self._setup_environment()
        self._load_data()
        self._initialize_arrays()
        
    # def _setup_environment(self):
    #     """Setup logging and environment variables."""
    #     logging.basicConfig(
    #         filename='my_script.log', 
    #         level=logging.INFO, 
    #         format='%(asctime)s - %(levelname)s - %(message)s'
    #     )
        
    #     # Set MOSEK license path
    #     mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
    #     os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
    #     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    def _setup_environment(self):
        """Setup logging and environment variables."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Remove all handlers associated with the root logger (if any)
        if logger.hasHandlers():
            logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler('my_script.log')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console (stream) handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Environment variables (keep these as before)
        mosek_license_path = r"C:\Users\kimjosy\Downloads\mosek\mosek.lic"
        os.environ["MOSEKLM_LICENSE_FILE"] = mosek_license_path
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        
    def _load_data(self):
        """Load and process the Gaussian data."""
        Gaus_info = np.load(self.file_path, allow_pickle=True)
        self.A = Gaus_info["A"]
        self.n, self.m = self.A.shape
        self.TE = Gaus_info["TE"].flatten()
        self.T2 = np.linspace(10, 200, self.m)
        
        # Generate regularization parameters
        self.Lambda = np.logspace(
            self.config.reg_param_lb, 
            self.config.reg_param_ub, 
            self.config.N_reg
        ).reshape(-1, 1)
        
    def _initialize_arrays(self):
        """Initialize arrays for simulation data."""
        # Calculate sigma and rps values
        self.rps = np.linspace(1.1, 4, self.config.nsigma).T
        self.config.nrps = len(self.rps)
        self.unif_sigma = np.linspace(2, 5, self.config.nsigma).T
        self.diff_sigma = np.column_stack((self.unif_sigma, 3 * self.unif_sigma))
        
        # Calculate T2 means for each rps
        self.T2mu = self._calc_T2mu()
        
        # Initialize noise arrays
        shape = (self.config.n_sim, self.config.nsigma, self.config.nrps, self.n)
        self.noise_arr = np.zeros(shape)
        self.stdnoise_data = np.zeros(shape)
        
        # Create target iterator for parallel processing
        self.target_iterator = [
            (a, b, c) 
            for a in range(self.config.n_sim) 
            for b in range(self.config.nsigma) 
            for c in range(self.config.nrps)
        ]
        
    def _calc_T2mu(self) -> np.ndarray:
        """Calculate T2 mean values for different peak separations."""
        mps = self.rps / 2
        T2_left = 40 * np.ones(len(self.rps))
        T2_right = T2_left * self.rps
        return np.column_stack((T2_left, T2_right))
    
    def _create_result_folder(self) -> str:
        """Create folder for saving results."""
        today = date.today()
        folder_name = f"MRR_1D_LocReg_Comparison_{today.strftime('%Y-%m-%d')}_nsim{self.config.n_sim}_SNR{self.config.SNR_value}"
        
        base_path = os.path.join("SimulationSets", "MRR", "SpanRegFig")
        full_path = os.path.join(os.getcwd(), base_path, folder_name)
        
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def _generate_ideal_model(self, iter_rps: int, sigma_i: np.ndarray) -> np.ndarray:
        """Generate the ideal/ground truth model."""
        T2mu_sim = self.T2mu[iter_rps, :]
        p = np.array([
            normsci.pdf(self.T2, mu, sigma) 
            for mu, sigma in zip(T2mu_sim, sigma_i)
        ])
        return (p.T @ np.ones(self.config.npeaks)) / self.config.npeaks
    
    def _add_noise(self, dat_noiseless: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Add noise to noiseless data."""
        if seed is not None:
            np.random.seed(seed)
            
        SD_noise = np.max(np.abs(dat_noiseless)) / self.config.SNR_value
        noise = np.random.normal(0, SD_noise, size=dat_noiseless.shape)
        return dat_noiseless + noise, noise, SD_noise
    
    def _minimize_oracle(self, Alpha_vec: np.ndarray, data_noisy: np.ndarray, g: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
        """Find oracle solution by minimizing error over regularization parameters."""
        nT2 = len(self.T2)
        OP_x_lc_vec = np.zeros((nT2, len(Alpha_vec)))
        OP_rhos = np.zeros(len(Alpha_vec))
        
        for j, alpha in enumerate(Alpha_vec):
            try:
                sol, _, _ = nonnegtik_hnorm(self.A, data_noisy, alpha, '0', nargin=4)
                if np.all(sol == 0):
                    raise ValueError("Zero vector detected")
            except Exception:
                # Fallback to CVXPY solver
                sol = self._solve_with_cvxpy(alpha, data_noisy)
                
            OP_x_lc_vec[:, j] = sol
            OP_rhos[j] = self._calculate_error(g, sol)
        
        min_index = np.argmin(OP_rhos)
        return (OP_x_lc_vec[:, min_index], 
                Alpha_vec[min_index][0], 
                OP_rhos[min_index], 
                min_index)
    
    def _solve_with_cvxpy(self, alpha: float, data_noisy: np.ndarray) -> np.ndarray:
        """Solve regularization problem using CVXPY."""
        lam_vec = alpha * np.ones(self.A.shape[1])
        A_reg = self.A.T @ self.A + np.diag(lam_vec)
        eps = 1e-2
        ep4 = np.ones(A_reg.shape[1]) * eps
        b = self.A.T @ data_noisy + self.A.T @ self.A @ ep4 + ep4 * lam_vec
        
        y = cp.Variable(self.A.shape[1])
        cost = cp.norm(A_reg @ y - b, 2)**2
        constraints = [y >= 0]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)
        
        sol = y.value - eps
        return np.maximum(sol, 0)
    
    def _calculate_error(self, true_signal: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate error between true and reconstructed signals."""
        if self.config.err_type == "WassScore":
            return wasserstein_distance(true_signal, reconstructed)
        else:
            true_norm = linalg_norm(true_signal)
            return linalg_norm(true_signal - reconstructed) / true_norm
    
    def _normalize_solution(self, solution: np.ndarray) -> np.ndarray:
        """Normalize solution using trapezoidal integration."""
        integral = np.trapz(solution, self.T2)
        return solution / integral if integral != 0 else solution
    
    def _run_all_methods(self, data_noisy: np.ndarray, IdealModel_weighted: np.ndarray, noise_norm: float) -> dict:
        """Run all regularization methods and return results."""
        results = {}
        
        # Standard methods
        results['f_rec_DP'], results['lambda_DP'] = discrep_L2(data_noisy, self.A, self.config.SNR_value, self.Lambda, noise_norm)
        results['f_rec_LC'], results['lambda_LC'] = Lcurve(data_noisy, self.A, self.Lambda)
        results['f_rec_GCV'], results['lambda_GCV'] = GCV_NNLS(data_noisy, self.A, self.Lambda)
        results['f_rec_GCV'] = results['f_rec_GCV'][:, 0]
        results['lambda_GCV'] = np.squeeze(results['lambda_GCV'])
        
        # Oracle method
        (results['f_rec_oracle'], 
         results['lambda_oracle'], 
         results['min_rhos'], 
         results['min_index']) = self._minimize_oracle(self.Lambda, data_noisy, IdealModel_weighted)
        
        # LocReg methods
        initial_lambda = results['lambda_GCV']  # Use GCV as initial
        maxiter = 50
        
        results['f_rec_LocReg'], results['lambda_LR'], _, _, _ = LocReg_Ito_mod(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        results['f_rec_LocReg1'], results['lambda_LR1'], _, _, _ = LocReg_Ito_mod_deriv(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        results['f_rec_LocReg2'], results['lambda_LR2'], _, _, _ = LocReg_Ito_mod_deriv2(
            data_noisy, self.A, initial_lambda, self.config.gamma_init, maxiter)
        
        # UPEN methods
        try:
            results['f_rec_upen'], results['lambda_upen'] = UPEN_Zama(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
            results['f_rec_upen1D'], results['lambda_upen1D'] = UPEN_Zama1st(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
            results['f_rec_upen0D'], results['lambda_upen0D'] = UPEN_Zama0th(
                self.A, data_noisy, IdealModel_weighted, noise_norm, 
                self.config.beta_0, self.config.Kmax, self.config.tol_lam)
        except Exception as e:
            print(f"UPEN methods failed: {e}")
            # Set default values
            for key in ['f_rec_upen', 'f_rec_upen1D', 'f_rec_upen0D']:
                results[key] = np.zeros_like(IdealModel_weighted)
            for key in ['lambda_upen', 'lambda_upen1D', 'lambda_upen0D']:
                results[key] = 0.0
        
        # Normalize all solutions
        solution_keys = [k for k in results.keys() if k.startswith('f_rec_')]
        for key in solution_keys:
            results[key] = self._normalize_solution(results[key].flatten())
            
        return results
    
    def _calculate_all_errors(self, results: dict, IdealModel_weighted: np.ndarray) -> dict:
        """Calculate all error metrics for the results."""
        errors = {}
        solution_keys = [k for k in results.keys() if k.startswith('f_rec_')]
        
        for key in solution_keys:
            method_name = key.replace('f_rec_', 'err_')
            errors[method_name] = self._calculate_error(IdealModel_weighted, results[key])
            
            # Calculate additional metrics
            diff = results[key] - IdealModel_weighted
            errors[method_name.replace('err_', 'bias_')] = np.mean(diff)
            errors[method_name.replace('err_', 'var_')] = np.var(diff)
            errors[method_name.replace('err_', 'MSE_')] = np.mean(diff**2)
            
        return errors
    
    def generate_single_estimate(self, params: Tuple[int, int, int]) -> pd.DataFrame:
        """Generate estimates for a single parameter combination."""
        iter_sim, iter_sigma, iter_rps = params
        
        # Get parameters for this iteration
        sigma_i = self.diff_sigma[iter_sigma, :]
        IdealModel_weighted = self._generate_ideal_model(iter_rps, sigma_i)
        
        # Generate noisy data
        dat_noiseless = self.A @ IdealModel_weighted
        if not self.config.preset_noise:
            dat_noisy, noise, stdnoise = self._add_noise(dat_noiseless, seed=iter_sim)
            self.noise_arr[iter_sim, iter_sigma, iter_rps, :] = noise
            self.stdnoise_data[iter_sim, iter_sigma, iter_rps, :] = stdnoise
        else:
            noise = self.noise_arr[iter_sim, iter_sigma, iter_rps, :]
            dat_noisy = dat_noiseless + noise
            stdnoise = self.stdnoise_data[iter_sim, iter_sigma, iter_rps, 0]
        
        noise_norm = np.linalg.norm(noise)
        
        # Run all methods
        results = self._run_all_methods(dat_noisy, IdealModel_weighted, noise_norm)
        
        # Calculate errors
        errors = self._calculate_all_errors(results, IdealModel_weighted)
        
        # Check if oracle is actually optimal
        oracle_err = errors['err_oracle']
        other_errs = [errors['err_LC'], errors['err_GCV'], errors['err_DP']]
        if not all(oracle_err <= err for err in other_errs):
            print(f"Warning: Oracle not optimal at iter {iter_sim}, sigma {iter_sigma}, rps {iter_rps}")
            logging.info(f"Warning: Oracle not optimal at iter {iter_sim}, sigma {iter_sigma}, rps {iter_rps}")
            return self._create_empty_dataframe(iter_sim, sigma_i, self.rps[iter_rps], IdealModel_weighted)
        
        # Create results dataframe
        return self._create_results_dataframe(
            iter_sim, sigma_i, self.rps[iter_rps], IdealModel_weighted, 
            results, errors
        )
    
    def _create_results_dataframe(self, iter_sim: int, sigma_i: np.ndarray, rps_val: float, 
                                IdealModel_weighted: np.ndarray, results: dict, errors: dict) -> pd.DataFrame:
        """Create a pandas DataFrame with all results."""
        data = {
            'NR': [iter_sim],
            'Sigma': [sigma_i],
            'RPS_val': [rps_val],
            'GT': [IdealModel_weighted]
        }
        
        # Add all results and errors to the dataframe
        for key, value in {**results, **errors}.items():
            data[key] = [value]
            
        return pd.DataFrame(data)
    
    def _create_empty_dataframe(self, iter_sim: int, sigma_i: np.ndarray, 
                              rps_val: float, IdealModel_weighted: np.ndarray) -> pd.DataFrame:
        """Create an empty dataframe for failed cases."""
        return pd.DataFrame({
            'NR': [iter_sim],
            'Sigma': [sigma_i],
            'RPS_val': [rps_val],
            'GT': [IdealModel_weighted]
        })
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the complete simulation."""
        logging.info("Starting simulation...")
        
        self.result_folder = self._create_result_folder()
        results_list = []
        
        if self.config.parallel:
            with mp.Pool(processes=os.cpu_count()) as pool:
                with tqdm(total=len(self.target_iterator)) as pbar:
                    for result in pool.imap_unordered(self.generate_single_estimate, self.target_iterator):
                        results_list.append(result)
                        pbar.update()
        else:
            for params in tqdm(self.target_iterator):
                result = self.generate_single_estimate(params)
                results_list.append(result)
        
        # Combine all results
        final_df = pd.concat(results_list, ignore_index=True)
        
        # Save results
        self._save_results(final_df)
        
        # Generate plots if requested
        if self.config.show:
            self._generate_plots(final_df)
        
        logging.info("Simulation completed.")
        return final_df
    
    def _save_results(self, df: pd.DataFrame):
        """Save simulation results to file."""
        today = date.today()
        filename = f"est_table_SNR{self.config.SNR_value}_iter{self.config.n_sim}_{today.strftime('%d%b%y')}.pkl"
        filepath = os.path.join(self.result_folder, filename)
        
        df.to_pickle(filepath)
        print(f"Results saved to: {filepath}")
        
        # Save noise arrays if not using preset noise
        if not self.config.preset_noise:
            np.save(os.path.join(self.result_folder, 'noise_arr.npy'), self.noise_arr)
            np.save(os.path.join(self.result_folder, 'stdnoise_data.npy'), self.stdnoise_data)
    
    def _generate_plots(self, df: pd.DataFrame):
        """Generate heatmap plots for visualization."""
        # Group and average results
        df['Sigma'] = df['Sigma'].apply(tuple)
        grouped = df.groupby(['Sigma', 'RPS_val']).agg({
            col: 'mean' for col in df.columns if col.startswith('err_')
        }).reset_index()
        
        # Create heatmaps (simplified version)
        self._create_error_heatmaps(grouped)
    
    def _create_error_heatmaps(self, grouped_df: pd.DataFrame):
        """Create heatmap visualizations of errors."""
        # This is a simplified version - you can expand based on your specific needs
        error_cols = [col for col in grouped_df.columns if col.startswith('err_')]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(error_cols[:8]):  # Plot first 8 error types
            if i < len(axes):
                # Reshape data for heatmap
                pivot_data = grouped_df.pivot(index='Sigma', columns='RPS_val', values=col)
                sns.heatmap(pivot_data, ax=axes[i], cmap='viridis', annot=True, fmt='.2e')
                axes[i].set_title(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_folder, 'error_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the simulation."""
    # Configuration
    config = SimulationConfig(
        n_sim=4,
        SNR_value=1000,
        parallel=False,  # Set to True for parallel processing
        show=True
    )
    
    # File path to the data
    file_path = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\num_of_basis_functions\lambda_16_SNR_1000_nrun_20_sigma_min_2_sigma_max_6_basis2_40110lmbda_min-6lmbda_max008Oct24.pkl"
    
    # Create simulator and run
    simulator = RegularizationSimulator(config, file_path)
    results_df = simulator.run_simulation()
    
    print(f"Simulation completed. Results shape: {results_df.shape}")
    return results_df

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    results = main()