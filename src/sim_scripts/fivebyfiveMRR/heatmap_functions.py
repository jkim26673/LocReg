from src.utils.load_imports.loading import *
from src.utils.load_imports.load_regmethods import *

# Setting up kernel matrix
n = 150
m = 200
TE = np.linspace(0.3, 400, n)
T2 = np.linspace(1, 200, m)
A = np.zeros((n, m))
dT = T2[1] - T2[0]

for i in range(n):
    for j in range(m):
        A[i, j] = np.exp(-TE[i] / T2[j]) * dT  # set up Kernel matrix

# Generate a sequence of Simulation
npeaks = 2
rps = np.linspace(1, 5, 5)
mps = rps / 2
nrps = len(rps)
T2_left = 20 * np.ones(nrps)
T2_mid = T2_left * mps
T2_right = T2_left * rps
T2mu = np.array([T2_left, T2_right]).T
# T2mu = np.array([T2_left, T2_mid, T2_right]).T
nsigma = 5
unif_sigma = np.linspace(2, 5, nsigma)
f_coef = np.ones(npeaks)

avg_MDL_err = np.zeros((nsigma, nrps))

SNR = 500
n_sim = 1

fig, axes = plt.subplots(nsigma, 5, figsize=(15, 15))

for i in range(nsigma):
    sigma_i = unif_sigma[i]
    for j in range(nrps):

        # setting up the distributions
        p = np.zeros((npeaks, m))
        T2mu_sim = T2mu[j]
        for ii in range(npeaks):
            p[ii, :] = np.exp(-0.5 * ((T2 - T2mu_sim[ii]) / sigma_i) ** 2) / (sigma_i * np.sqrt(2 * np.pi))

        IdealModel_weighted = p.T.dot(f_coef)
        dat_noiseless = A.dot(IdealModel_weighted)
        dat_noisy = dat_noiseless + max(abs(dat_noiseless)) / SNR * np.random.randn(len(TE))
        IdealModel_weighted = IdealModel_weighted / max(abs(dat_noisy))
        dat_noisy = dat_noisy / max(abs(dat_noisy))

        print(nsigma * i + j)

        axes[i, j].plot(T2, IdealModel_weighted, linewidth=1, color='black')
        axes[i, j].set_xlim(0, 200)
        axes[i, j].set_ylim(-0.1, 1.1)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        drawnow

plt.show()
