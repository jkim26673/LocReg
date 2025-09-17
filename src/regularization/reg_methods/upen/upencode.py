import matlab.engine
import numpy as np
import scipy
import pandas as pd
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\Users\kimjosy\Downloads\LocReg_Regularization-1\MERA', nargout=0)

# def initialize_upen(TE, T2):
#     numT2 = len(T2)
#     K1 = numT2 + 1
#     mux = 0.001
#     dQ = TE[1] - TE[0]
#     # Encapsulate the values in a dictionary
#     result = {
#         "numT2": numT2,
#         "K1": K1,
#         "mux": mux,
#         "dQ": dQ
#     }
#     return result

def H_mat(K1):
    """
    Description: Oracle selection method for regularization:

    :param power_pellet_active: bool - does the player have an active power pellet?
    :param touching_ghost: bool - is the player touching a ghost?
    :return: bool - can a ghost be eaten?

    Test Example:
    """
    #import H
    H = -2 * np.eye(K1) + np.diag(np.ones(K1 - 1), k=1) + np.diag(np.ones(K1 - 1), k=-1)
    # Step 2: Set the first and last rows to zero
    H[0, :] = 0
    H[-1, :] = 0
    return H

def decompose_Gmat(G):
  """
  Description: Oracle selection method for regularization:

  :param power_pellet_active: bool - does the player have an active power pellet?
  :param touching_ghost: bool - is the player touching a ghost?
  :return: bool - can a ghost be eaten?

  Test Example:
  """
   G_matlab = matlab.double(G.tolist())
   U,sig,V= eng.svd(G_matlab, "econ", nargout = 3)
   U = np.array(U)
   sig = np.array(sig)
   V = np.array(V)
   s = np.linalg.matrix_rank(sig)
   Ur = U[:,:s]
   Ar = sig[:s,:s] @ (V[:,:s].T)
   return G, Ur, Ar

def upen_iter(param_setup, lam, S, R ,Dm, Dx):
  """
  Description: Oracle selection method for regularization:

  :param power_pellet_active: bool - does the player have an active power pellet?
  :param touching_ghost: bool - is the player touching a ghost?
  :return: bool - can a ghost be eaten?

  Test Example:
  """
    # Define betaL, beta0, betaP, and betaC
    betaL = [1e-5, 0.6, 0.3]
    beta0, betaP, betaC = betaL[0], betaL[1], betaL[2]
    betaA = 0.5
    G = param_setup["G"]
    # Initialize Sq and Rq
    Sq = S
    Rq = R
    # Loop for 5 iterations
    for k in range(5):
        # Calculate p and c
        p = np.concatenate(([0], (Sq[2:] - Sq[:-2])**2, [0]))
        c = np.concatenate(([0], np.diff(Sq, 2)**2, [0]))
        # Calculate cm and ck
        # cm = np.maximum(np.stack([c, np.roll(c, 1, axis=0), np.roll(c, -1, axis=0)]), axis=1)
        c_shifted_down = np.concatenate(([0], c[:-1]))  # Shifted down, appending 0 at the top
        c_shifted_up = np.concatenate((c[1:], [0]))    # Shifted up, appending 0 at the bottom

        # Stack the three versions together
        c_stack = np.vstack([c, c_shifted_down, c_shifted_up]).T  # Shape (201, 3)
        # Find the row-wise maximum
        cm = np.max(c_stack, axis=1)
        ck = betaP * (param_setup["dQ"]**2) * p + betaC * cm
        
        # Calculate Ck and Hm
        Ck = np.sqrt(np.sum(cm) / (beta0 + ck))

        gk = Sq
        # Ak = np.sqrt(np.sum(cm) / (beta0 + betaA*(gk**2)))
        Ak = np.sum(cm) / (beta0 + betaA*(gk**2))
        lams = Ak

        Hm = param_setup["H"] * (Ck[:, np.newaxis] * np.ones((1, len(Ck))))
        
        # Update Am
        Am = np.vstack([param_setup["Ar"], np.sqrt(lam) * Hm])
        # Solve the non-negative least squares problem using scipy's nnls
        Am_matlab = matlab.double(Am.tolist())  # Convert NumPy array to MATLAB double
        Dm_matlab = matlab.double(Dm.tolist())  # Convert NumPy array to MATLAB double

        Sq = np.array(eng.nnlsLH(Am_matlab, Dm_matlab, "nnlsmex", nargout=1)).flatten()
        # Calculate Pq and Rq
        S_reshaped = np.reshape(S, (G.shape[1], param_setup["A2"]))
        Pq = np.dot(G, S_reshaped)
        Rq = Pq.flatten() - Dx.flatten()
    
    # Return final results
    return Sq, Rq, Hm, lams

# def upen_initialize(TE,T2,G):
#   numT2 = len(T2)
#   K1 = numT2 + 1
#   mux = 0.001
#   dQ = TE[1] - TE[0]
#   H = H_mat(K1)
#   G, Ur, Ar = decompose_Gmat(G)
#   Dr = Ur.T @ Dt
#   Ar_matlab = matlab.double(Ar.tolist())  # Convert NumPy array to MATLAB double


def upen_param_setup(TE, T2, G, d):
  """
  Description: Oracle selection method for regularization:

  :param power_pellet_active: bool - does the player have an active power pellet?
  :param touching_ghost: bool - is the player touching a ghost?
  :return: bool - can a ghost be eaten?

  Test Example:
  """
  numT2 = len(T2)
  K1 = numT2
  # K1 = numT2 + 1
  lam = 1e-3
  dQ = TE[1] - TE[0]
  H = H_mat(K1)
  G, Ur, Ar = decompose_Gmat(G)
  A2 = 1
  result = {
      "numT2": numT2,
      "K1": K1,
      "lam": lam,
      "dQ": dQ,
      "H": H,
      "G": G,
      "Ur": Ur,
      "Ar": Ar,
      "A2": A2
  }
  return result

def upen_setup(param_setup, D, lambdas, inputlam = True):
  """
  Description: Oracle selection method for regularization:

  :param power_pellet_active: bool - does the player have an active power pellet?
  :param touching_ghost: bool - is the player touching a ghost?
  :return: bool - can a ghost be eaten?

  Test Example:
  """
  if inputlam == True:
     lam = lambdas
  else:
     lam = param_setup["lam"]
  Dr = param_setup["Ur"].T @ D  
  Ar_matlab = matlab.double(param_setup["Ar"].tolist())  # Convert NumPy array to MATLAB double
  Dt_matlab = matlab.double((Dr).tolist())  # Convert NumPy array to MATLAB double
  # Call nnlsLH function
  Dt_matlab = eng.transpose(Dt_matlab)
  # Ar_matlab = np.array(Ar_matlab)
  # Dt_matlab = np.array(Dt_matlab).flatten()
  S = np.array(eng.nnlsLH(Ar_matlab, Dt_matlab, "nnlsmex", nargout=1)).flatten()
  S_reshaped = np.reshape(S, (param_setup["G"].shape[1], param_setup["A2"])).flatten()
  P = np.dot(param_setup["G"], S_reshaped)
  R = P-D
  Er = np.linalg.norm(R)**2; # sum square error
  Er0 = Er; # sum square error of unregularized fit
  dc = S[-1]; # dc-offset term
  zeros_column = np.zeros((param_setup["H"].shape[0], 1))
  # Concatenate Dr and zeros_column vertically
  Dm = np.vstack([Dr.reshape(-1,1), zeros_column])
  Sq, Rq, Hm, lams= upen_iter(param_setup, lam, S, R, Dm, D)
  return Sq, Rq, Hm, lams


# G, Ur, Ar = decompose_Gmat(G)
# Dr = Ur.T @ Dt
# Dt = np.arange(1,3).reshape(-1,1)
# Dr = Dt
# Ar_matlab = matlab.double(Ar.tolist())  # Convert NumPy array to MATLAB double
# Dt_matlab = matlab.double(Dt.tolist())  # Convert NumPy array to MATLAB double

# # Call nnlsLH function
# S = np.array(eng.nnlsLH(Ar_matlab, Dt_matlab, "nnlsmex", nargout=1)).flatten()
# A2 = 1
# # S = eng.nnlsLH(Ar,Dr,nnlscode);
# # Reshape S to the correct shape (size of Amat.A, size of Amat.A2)
# S_reshaped = np.reshape(S, (G.shape[1], A2)).flatten()
# # Perform the matrix multiplication equivalent to the MATLAB expression
# P = np.dot(G, S_reshaped)
# R = P-Dt
# Er = np.linalg.norm(R)**2; # sum square error
# Er0 = Er; # sum square error of unregularized fit
# dc = S[-1]; # dc-offset term


