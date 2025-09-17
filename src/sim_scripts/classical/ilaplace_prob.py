from src.utils.load_imports.load_classical import *
from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *
n = 100
nT2 = n
T2 = np.linspace(0,1000,n)
TE = T2

G, data_noiseless, g, _ = i_laplace(n, example=1 , nargin=1)

U,s,V = csvd(G,tst = None, nargin = 1, nargout = 3)
SNR = 30
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-10,5,40)
nLambda = len(Lambda_vec)

nrun = 10
com_vec_DP = np.zeros(nrun)
com_vec_GCV = np.zeros(nrun)
com_vec_LC = np.zeros(nrun)
com_vec_locreg = np.zeros(nrun)
res_vec_DP = np.zeros((n,nrun))
res_vec_GCV = np.zeros((n,nrun))
res_vec_LC = np.zeros((n,nrun))
res_vec_locreg = np.zeros((n,nrun))

eps1 = 1e-1
ep_min = 1e-2
eps_cut = 1.2
eps_floor = 1e-4
feedback = True
exp = 0.5

for i in range(nrun):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    # L-curve
    #reg_corner,rho,eta = l_curve(U,s,data_noisy)
    lambda_LC,rho,eta,_ = l_curve(U,s,data_noisy, method = None, L = None, V = None, nargin = 3, nargout = 3)
    # f_rec_LC,_,_ = tikhonov(U,s,V,data_noisy,lambda_LC, nargin=5, nargout=1)
    f_rec_LC, lambda_LC = tikhonov_vec(U, s, V, data_noisy,lambda_LC, x_0 = None, nargin = 5)
    com_vec_LC[i] = norm(g - f_rec_LC)
    res_vec_LC[:,i] = f_rec_LC
    # %% GCV
    reg_min,_,reg_param = gcv(U,s,data_noisy, method = 'Tikh', nargin = 3, nargout = 3)
    # f_rec_GCV,_,_ = tikhonov(U,s,V,data_noisy,reg_min, nargin=5, nargout=1)
    f_rec_GCV, lambda_GCV = tikhonov_vec(U, s, V, data_noisy,reg_min, x_0 = None, nargin = 5)
    com_vec_GCV[i] = norm(g - f_rec_GCV)
    res_vec_GCV[:,i] = f_rec_GCV
    
    # %% DP
    delta = norm(noise)*1.05
    x_delta,lambda_DP = discrep(U,s,V,data_noisy,delta, x_0= None, nargin = 5)
    # f_rec_DP,_,_ = tikhonov(U,s,V,data_noisy,lambda_DP, nargin=5, nargout=1)
    f_rec_DP, lambda_DP = tikhonov_vec(U, s, V, data_noisy,reg_min, x_0 = None, nargin = 5)
    com_vec_DP[i] = norm(g - f_rec_DP)
    res_vec_DP[:,i] = f_rec_DP
    
    # %% locreg
    x0_ini = f_rec_LC
    # %     lambda_ini = reg_corner;
    ep1 = 1e-2
    # % 1/(|x|+ep1)
    ep2 = 1e-1
    # % norm(dx)/norm(x)
    ep3 = 1e-2
    # % norm(x_(k-1) - x_k)/norm(x_(k-1))
    ep4 = 1e-4 
    # % lb for ep1
    maxiter = 75
    gamma_init = 10

    # f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)
    f_rec_locreg, lambda_locreg = LocReg_Ito_mod(data_noisy, G, lambda_LC, gamma_init, maxiter)
    # f_rec_locreg, lambda_locreg, gamma_new1, _, _ = LocReg_Ito_UC_3(data_noisy, G, lambda_LC, gamma_init, maxiter, eps1, ep_min, eps_cut, eps_floor, exp, feedback)
    LR_Ito_lams3 = lambda_locreg
    # com_vec_ItoLR3[i] = norm(g - LR_mod_rec3)

    # f_rec_locreg, lambda_locreg = LocReg_unconstrained(data_noisy, G, x0_ini, lambda_LC, ep1, ep2, ep3)
    com_vec_locreg[i] = norm(g - f_rec_locreg)
    res_vec_locreg[:,i] = f_rec_locreg

print('The mean error for DP is', str(np.mean(com_vec_DP)))
print('The mean error for L-curve is', str(np.mean(com_vec_LC)))
print('The mean error for GCV is', str(np.mean(com_vec_GCV)))
print('The mean error for locreg is', str(np.mean(com_vec_locreg)))

print('The SD for DP is', str(np.std(com_vec_DP)))
print('The SD for L-curve is', str(np.std(com_vec_LC)))
print('The SD for GCV is', str(np.std(com_vec_GCV)))
print('The SD for locreg is', str(np.std(com_vec_locreg)))

rid_GCV = np.where(com_vec_GCV >= 100)[0]
res_vec_GCV = np.delete(res_vec_GCV, rid_GCV, axis=1)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(T2, g, linewidth=1.5 , color = "black")
plt.plot(T2, f_rec_DP, color = "blue")
plt.title('DP')

plt.subplot(2, 2, 2)
plt.plot(T2, g, linewidth=1.5, color = "black")
plt.plot(T2, f_rec_LC, color = "orange")
plt.title('L curve')

plt.subplot(2, 2, 3)
plt.plot(T2, g, linewidth=1.5, color = "black")
plt.plot(T2, f_rec_GCV, color = "green")
plt.title('GCV')

plt.subplot(2, 2, 4)
plt.plot(T2, g, linewidth=1.5, color = "black")
plt.plot(T2, f_rec_locreg, color = "red")
plt.title('LocReg')

plt.tight_layout()
plt.savefig('ilaplace_prob_nrun_10_08_22_23.png')  # You can specify the desired filename and format
plt.show()