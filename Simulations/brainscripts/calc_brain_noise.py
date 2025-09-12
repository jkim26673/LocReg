import matplotlib.pyplot as plt
import scipy.io
import pickle
import random
import numpy as np
from scipy.optimize import nnls
unmasked_filepath = "/home/kimjosy/LocReg_Regularization-1/data/brain/masks/new_mask.mat"
mask_array = scipy.io.loadmat(unmasked_filepath)["new_BW"]
unfiltered_estimates_path = "/home/kimjosy/LocReg_Regularization-1/data/brain/braindata/mew_cleaned_brain_data_unfiltered.mat"
unfiltered_arr = scipy.io.loadmat(unfiltered_estimates_path)["brain_data"]

dTE = 11.3
n = 32
TE = dTE * np.linspace(1,n,n)
m = 150
T2 = np.linspace(10,200,m)
A= np.zeros((n,m))
dT = T2[1] - T2[0]
for i in range(n):
    for j in range(m):
        A[i,j] = np.exp(-TE[i]/T2[j]) * dT

num_signals = 5
coord_pairs = set()
for i in range(num_signals):
    x = random.randint(155, 160)
    y = random.randint(155, 160)
    coord_pairs.add((x,y))
coord_pairs = list(coord_pairs)

signals = []
for (x_coord, y_coord) in coord_pairs:
    mask_value = mask_array[x_coord,y_coord]
    signal = unfiltered_arr[x_coord,y_coord,:]
    sol1 = nnls(A, signal, )[0]
    factor = np.sum(sol1) * dT
    signal = signal/factor
    normalized_signal = signal
    signals.append(normalized_signal)
    print((x_coord, y_coord))
    print("mask_value", mask_value)

# Convert signals to a NumPy array for easier manipulation
signals = np.array(signals)
mean_sig = np.mean(signals, axis=0)
squared_diff = (signals - mean_sig) ** 2
sum_squared_diff = np.sum(squared_diff, axis=0)
noise = np.sqrt(sum_squared_diff)
random_signs = np.random.choice([-1, 1], size=noise.shape)
noise = noise * random_signs
print("noise", noise)

plt.figure()
plt.imshow(mask_array)
for idx, (x_coord, y_coord) in enumerate(coord_pairs):
    plt.annotate(f"{x_coord}, {y_coord}", xy = (x_coord, y_coord), xytext=(0.5, 0.5),textcoords='figure points',
                arrowprops=dict(arrowstyle="->"))
    color_map = ["red", "blue", "green", "orange", "gold"]
    plt.scatter(x_coord, y_coord, c=color_map[idx % len(color_map)], marker="o")
plt.savefig("test_fig.png")
plt.close()