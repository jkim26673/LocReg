import numpy as np
def gen_noiseset(SNR,data_noiseless,nrun):
    noise_set = []
    for i in range(nrun):
        SD_noise= 1/SNR*max(abs(data_noiseless))
        noise = np.random.normal(0,SD_noise, data_noiseless.shape)
        noise_set.append(noise)
    return np.array(noise_set)