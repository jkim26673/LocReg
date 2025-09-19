# import matlab.engine
# import numpy as np
# import scipy
# import pandas as pd
# eng = matlab.engine.start_matlab()
# eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\ZamaUPEN\1D_test', nargout=0)

from oct2py import Oct2Py
import numpy as np
import scipy
import pandas as pd

# Start Octave session
oc = Oct2Py()

# Add your Octave path (adjust as needed)
oc.addpath(r'/Users/joshuakim/Downloads/Coding_Projects/LocReg/LocReg/tools/zama_upen/1D_test')

def UPEN_Zama(A, b, gt, noise_norm, beta_0, Kmax, tol_lam):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Call the MATLAB function UPenMMmL2nn and only get the first and last output (xpwL2_nn and Lam)
    A = matlab.double(A.tolist())
    b = matlab.double(b.tolist())
    gt = matlab.double(gt.tolist())
    reshaped_b = eng.reshape(b, [], 1)  # Reshape to a column vector
    reshaped_gt = eng.reshape(gt, [], 1)  # Reshape to a column vector
    # gt = matlab.double(gt.tolist())
    result = eng.UPenMMmL2nn(A, reshaped_b, reshaped_gt, noise_norm, beta_0, Kmax, tol_lam, nargout=9)
    # Extract the specific outputs (xpwL2_nn and Lam)
    xpwL2_nn = np.array(result[0]).flatten()  # First output (solution)
    Lam = np.array(result[5]).flatten()  # 5 output (Lambda change)
    # Return the results to Python
    # xpwL2_nn = np.array(xpwL2_nn.flatten())
    return xpwL2_nn, Lam

def UPEN_Zama1st(A, b, gt, noise_norm, beta_0, Kmax, tol_lam):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Call the MATLAB function UPenMMmL2nn and only get the first and last output (xpwL2_nn and Lam)
    A = matlab.double(A.tolist())
    b = matlab.double(b.tolist())
    gt = matlab.double(gt.tolist())
    reshaped_b = eng.reshape(b, [], 1)  # Reshape to a column vector
    reshaped_gt = eng.reshape(gt, [], 1)  # Reshape to a column vector
    # gt = matlab.double(gt.tolist())
    result = eng.UPennn1st(A, reshaped_b, reshaped_gt, noise_norm, beta_0, Kmax, tol_lam, nargout=9)
    # Extract the specific outputs (xpwL2_nn and Lam)
    xpwL2_nn = np.array(result[0]).flatten()  # First output (solution)
    Lam = np.array(result[5]).flatten()  # 5 output (Lambda change)
    # Return the results to Python
    # xpwL2_nn = np.array(xpwL2_nn.flatten())
    return xpwL2_nn, Lam

def UPEN_Zama0th(A, b, gt, noise_norm, beta_0, Kmax, tol_lam):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    # Call the MATLAB function UPenMMmL2nn and only get the first and last output (xpwL2_nn and Lam)
    A = matlab.double(A.tolist())
    b = matlab.double(b.tolist())
    gt = matlab.double(gt.tolist())
    reshaped_b = eng.reshape(b, [], 1)  # Reshape to a column vector
    reshaped_gt = eng.reshape(gt, [], 1)  # Reshape to a column vector
    # gt = matlab.double(gt.tolist())
    result = eng.UPennn0th(A, reshaped_b, reshaped_gt, noise_norm, beta_0, Kmax, tol_lam, nargout=9)
    # Extract the specific outputs (xpwL2_nn and Lam)
    xpwL2_nn = np.array(result[0]).flatten()  # First output (solution)
    Lam = np.array(result[5]).flatten()  # 5 output (Lambda change)
    # Return the results to Python
    # xpwL2_nn = np.array(xpwL2_nn.flatten())
    return xpwL2_nn, Lam