from regu.csvd import csvd
import numpy as np
import matplotlib.pyplot as plt
from Utilities_functions.tikhonov_vec import tikhonov_vec
from numpy.linalg import norm
from scipy.special import airy,gamma,itairy, gammaln, entr,tklmbda,expit


# def LocReg_unconstrained(data_noisy, G, x0_ini, lambda_ini, ep1, ep2, ep3):
#     U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
#     UD_fix = U.T @ data_noisy
#     W = V @ np.diag(UD_fix)

#     #Why are we multiplying by 2
#     prev_x = 2*x0_ini
#     #prev_x = 0.5 * x0_ini
#     x_DP = np.zeros(len(s))
#     for i in range(len(s)):
#         x_DP[i] = s[i] / (s[i]**2 + lambda_ini**2)
#     f_rec_DP = W @ x_DP
#     p0 = f_rec_DP
#     #Standard deviation of the noise residual
#     SD_noise = np.std(data_noisy - G @ x0_ini)
#     # x0_ini[x0_ini < 0] = 0
#     lambda_approx = 0.1 / (p0 + 1)
#     lambda_approx[:10] = 1e-2
#     p1 = W @ (s / (s**2 + lambda_approx))




    

#     return f_rec_logreg, lambda_locreg
        



def LocReg_unconstrained_NE(data_noisy, G, x0_ini, lambda_ini, ep1, ep2, ep3):
    U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
    
    def relu(x):
	    return np.max(np.zeros_like(x), x)
    #Why are we multiplying by 2
    prev_x = 2*x0_ini
    #prev_x = 0.5 * x0_ini
    
    #Plot the vector of the lambda initial values
    lambda_ini = lambda_ini * (np.ones(G.shape[1]))

    #Standard deviation of the noise residual
    SD_noise = np.std(data_noisy - G @ x0_ini)
    # x0_ini[x0_ini < 0] = 0

    while True:
        plt.figure(1)
        plt.plot(x0_ini)
        plt.draw()
        #print("x0_ini:", x0_ini)
        # print("SD_noise:", SD_noise)

        #What happens if x0_ini is a scalar ; median(x0_ini) and thus dlambda1 is scalar

        # dlambda1 = (SD_noise) * (np.exp(-(x0_ini ** 100)))
        # dlambda1 = SD_noise / (SD_noise * x0_ini**5 + ep1)
        # dlambda1 = (x0_ini/(np.sqrt(2*np.pi))) / np.exp((-(x0_ini)/(2*(SD_noise**2))))
        # x0_ini / (np.max(x0_ini) - np.min(x0_ini))
        # dlambda1 = np.std(x0_ini) / np.sqrt(np.exp(np.sin(x0_ini) / np.cos(x0_ini)))
        # dlambda1 =  x0_ini / (1 + np.sqrt(x0_ini + max(x0_ini)) + 3*x0_ini**4 - np.log(x0_ini + max(x0_ini)) - np.exp(-x0_ini))
        
        #dlambda1 = SD_noise /(x0_ini**2 + np.log(x0_ini**2 + ep1) + ep1)
        #0.5157404027667268(heat_prob)


        
        #dlambda1 = SD_noise / (1 / (x0_ini**2) + np.log(x0_ini + max(x0_ini)) + ep1)
        
        #dlambda1 = SD_noise / (np.sin(x0_ini) + np.cos(x0_ini) + ep1)
        #0.22126587526805994 for heat prob
        #0.5667058481269003 for gravity prob
        # dlambda1 = SD_noise / (np.exp(-(x0_ini**3 + ep1)))

        #Figures;
        # pos_x0_ini = x0_ini
        # pos_x0_ini[pos_x0_ini < 0] = 0 
        
        # print(pos_x0_ini)
        # print(pos_x0_ini.shape)
        # dlambda1 = SD_noise / (x0_ini + ep1)
        # dlambda1 = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ G.T) + np.linalg.norm(x0_ini)))/((x0_ini) + ep1)
        # dlambda1 = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ G.T) + np.linalg.norm(G.T @ G @ x0_ini)  + np.linalg.norm(G.T @ G @ x0_ini)**2))/((x0_ini) + ep1)
        # dlambda1 = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ G.T) + np.linalg.norm(G.T @ G @ x0_ini)  + np.linalg.norm(G.T @ G @ x0_ini)**2))/((x0_ini) + ep1)
        dlambda1 = (np.linalg.norm(G.T @ data_noisy, 2)/ (np.linalg.norm(G.T @ G @ G.T, 2)))/ ((x0_ini) + ep1)
        # dlambda1 = (np.linalg.norm(G.T @ data_noisy)/ (np.linalg.norm(G.T @ G @ G.T) + ep1))/(x0_ini + ep1)
        #dlambda1 = (SD_noise * (-expit(x0_ini**2)) * x0_ini / (x0_ini**2 * np.exp(-x0_ini) + ep1)) 
        #0.3087072443987112 for gravity prob
        #0.6452586539526532 for heat prob


        #dlambda1 = SD_noise / (x0_ini + ep1)
        # dlambda1 = SD_noise / (x0_ini**10 + ep1)
        # dlambda1 = (x0_ini/(np.sqrt(2*np.pi))) / np.exp((-(x0_ini)/(2*(SD_noise**2))))

        dlambda1 = 1 * np.median(lambda_ini) * (dlambda1 - np.median(lambda_ini)) / (np.max(dlambda1) - np.min(dlambda1) + ep1)
        #Changed the center to be at the median and not the minimum.

        #Why are we scaling by mean (lambda_ini), maybe median is better?
        #Why centering around min (dlambda1); if dlambda1 is constant, then np.min is fixed

        #dlambda1 = 1 * np.mean(lambda_ini)* ((dlambda1) - np.median(dlambda1))/ (np.max(dlambda1) - np.min(dlambda1))
        # print("dlambda1:", dlambda1)
        #What if its a scalar multiply?
        lambda1 = (lambda_ini.flatten() + dlambda1) 
        #lambda1 = lambda_ini.flatten() * dlambda1 + lambda_ini.flatten()
        x0_ini = tikhonov_vec(U, s, V, data_noisy, lambda1, x_0 = None, nargin = 5)
        curr_noise = data_noisy - G @ x0_ini
        delta_p = tikhonov_vec(U, s, V, curr_noise, lambda1, x_0 = None,  nargin = 5)
        
        x0_ini = x0_ini + delta_p
        if norm(x0_ini - prev_x)/norm(prev_x) < ep3:
            f_rec_logreg = x0_ini
            lambda_locreg = lambda1.copy()
            break
        
        ep1 = ep1/1.1
        prev_x = x0_ini

    return f_rec_logreg, lambda_locreg





