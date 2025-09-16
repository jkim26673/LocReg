# import sys
# sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
# import numpy as np
# # from regu.baart import baart
# # from regu.foxgood import foxgood
# # from regu.phillips import phillips
# # from regu.csvd import csvd
# # from regu.deriv2 import deriv2
# # from regu.gravity import gravity
# # from regu.heat import heat
# # from regu.shaw import shaw
# from numpy.linalg import norm
# import matplotlib.pyplot as plt
# from regu.l_curve import l_curve
from utils.load_imports.load_classical import *
import numpy as np
from math import factorial
# from regu.discrep import discrep
import numpy as np
from scipy.interpolate import lagrange
# from pykalman import KalmanFilter, UnscentedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.signal import gaussian
from scipy.ndimage import convolve1d

n = 1000
nT2 = n
T2 = np.linspace(-np.pi/2,np.pi/2,n)
TE = T2
G,data_noiseless,g = baart(n, nargout=3)
# G,data_noiseless,g = phillips(n, nargout =3)
#G,data_noiseless,g = phillips(n, nargout=3)
unit_vector = 1000
NR = 1000
U,s,V = csvd(G, tst = None, nargin = 1, nargout = 3)
SNR = 1000
SD_noise= 1/SNR*max(abs(data_noiseless))
Lambda_vec = np.logspace(-5,5,40)
nLambda = len(Lambda_vec)

data_noisy_vec = []
gt_lambda = []
lambda_approx_vec = []
W_vec = []
p0_vec = []
s_vals = []
for i in range(NR):
    noise = np.random.normal(0,SD_noise, data_noiseless.shape)
    data_noisy = data_noiseless + noise
    data_noisy_vec.append(data_noisy)
    UD_fix = U.T @ data_noisy
    W = V @ np.diag(UD_fix)
    lambda_result = s / (np.linalg.solve(W, g)) - s**2
    gt_lambda.append(lambda_result)
    W_vec.append(W)
# DP
    delta = norm(noise)*1.05
    f_rec_DPA,lambda_DP = discrep(U,s,V,data_noisy,delta,  x_0= None, nargin = 5)
    x_DP = np.zeros(len(s))
    for i in range(len(s)):
        x_DP[i] = s[i] / (s[i]**2 + lambda_DP**2)
    f_rec_DP = W @ x_DP
    p0 = f_rec_DP
    p0_vec.append(p0)
    lambda_approx = 0.1 / (p0 + 1)
    s_vals.append(s)
    lambda_approx_vec.append(lambda_approx)

gt_lambda = np.array(gt_lambda)
data_noisy_vec = np.array(data_noisy_vec)
p0_vec = np.array(p0_vec)
s_vals = np.array(s_vals)
# W_vec = np.array(W_vec)
chunk_size = len(W_vec) // 10
split_arrays = [W_vec[i:i+chunk_size] for i in range(0, len(W_vec), chunk_size)]
batches_W = split_arrays
lambda_approx_vec = np.array(lambda_approx_vec)

#Split the NR into batches of 10 for both noisy data and the ground truth lambdas
# batches_noisy_data = (np.array_split(data_noisy_vec, 10))
# batches_lambda = (np.array_split(gt_lambda, 10))
# batches_approx = (np.array_split(lambda_approx_vec, 10))
# batches_svals = (np.array_split(s_vals, 10))
# batches_p0 = (np.array_split(p0_vec, 10))
# #Ground truth
lambda_result = s / (np.linalg.solve(W, g)) - s**2

batches_noisy_data = np.array(batches_noisy_data)
batches_lambda = np.array(batches_lambda)
#We want to predict the lambdas for each index position for all 10 lambdas. We will split 100 NR for the each position lambda. 
#We repeat this 10 times for each lambda position for 1000 NR
#For 1000 NR: i get the data_noisy values for 1000 NR. I split the data_noisy values into groups of ten. 
# I also keep track of the lambda ground truth values for each NR
#For the random forest, I indidcate the k index position that I am interested in and run 100 NR for each index.
# I report the results and compare with ground truth lambdas

# Calculate lambda_approx
lambda_approx = 0.1 / (p0 + 1)

#Try a machine learning algorithim to predict lambdas
# One possible approach is to use the available data, such as data_noisy and W,
# to train a regression model that can approximate lambda_result. Here's a general outline of the steps you can follow:


# Prepare the Data:

# Ensure that data_noisy and W are properly preprocessed and in suitable formats for training the regression model.
# Define the Training Set:
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV


def random_forest(index, data_batch, W, gt_lambda_batch):
    #k is the index of the lambdas of interest (1-10)
    #data_source should be data_noisy
    #data_vector should be the data_noisy_vec
    # k = 990
    data_set = data_batch
    #Randomly select 8 for

    # Create a training dataset by selecting a subset of the available data
    # data_subset = data_set[:k]
    data_batch = data_noisy_vec
    data_subset = data_batch
    gt_lambda_batch = gt_lambda
    
    k = 1000
    W_subset = W[:k, :]

    
    data_subset = data_noisy_vec
    # W_subset = W

    # Train a random forest
    # X = np.hstack((data_subset.reshape(-1, 1), W_subset, s_subset.reshape(-1,1)))  # Add singular values 's' to the input features
    X = np.hstack((data_subset.reshape(-1,1), W.reshape(-1,1)))  # Add singular values 's' to the input features

    # Reshape X to match the expected shape of (samples, features)
    X = np.reshape(X, (X.shape[0], -1))

    y = np.hstack((gt_lambda_batch.reshape(-1,1)))
    y = np.reshape(y, (y.shape[0], -1)).ravel()


    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest regression model
    
    
    rf_model = RandomForestRegressor(n_estimators= 100, random_state= 42)
    # Train the model
    rf_model.fit(X_train,y_train)
    # rf_model.fit(X_train, y_train)
    # Evaluate the model on the validation set
    y_pred = rf_model.predict(X_val)
    y_pred_reshape = y_pred.reshape(int(len(y_pred)/1000),1000)
    pred_lam_1 = y_pred_reshape[:,:5]
    gt_lam = y_val.reshape(int(len(y_val)/1000),1000)
    first_gt_lam = gt_lam[:,:5]


    mse = np.mean((pred_lam_1 - first_gt_lam) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_val))
    r2 = rf_model.score(X_val, y_val)

    # new_data_subset = data_set[9:-1]
    # # new_W_subset = W[0, :]

    # # new_data_subset = data_set[data_set != data_subset]
    # # new_W_subset = W[W != W_subset].reshape(W.shape[0]-1,W.shape[1])
    # # new_s = p0[k-1:-1]

    # # Use the trained model for prediction
    # new_X = np.hstack((new_data_subset.reshape(-1,1)))  # Add singular values to the new data
    # # new_X = np.hstack(new_data_subset)  # Add singular values to the new data
    # new_X = np.reshape(new_X, (new_X.shape[0], -1))
    # new_X = np.array(new_X)
    # predictions = rf_model.predict(new_X)
    # lams = gt_lambdas[0]

    return predictions, mse, mae,r2

#Hyper paramterize
from skopt import BayesSearchCV
n_iter = 70
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
# bootstrap = [True, False]
# Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
import tqdm

rf = RandomForestRegressor()
reg_bay = BayesSearchCV(estimator = rf, search_spaces = random_grid, n_iter = n_iter, cv = 5, n_jobs = 8, error_score='raise', random_state = 123)
model_bay = reg_bay.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_val, y_val)
y_pred = rf_model.predict(X_val)
y_pred_reshape = y_pred.reshape(int(len(y_pred)/1000),1000)
pred_lam_1 = y_pred_reshape[:,:5]
gt_lam = y_val.reshape(int(len(y_val)/1000),1000)
first_gt_lam = gt_lam[:,:5]


rf_random= RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 10, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    y_pred_reshape = predictions.reshape(int(len(predictions)/1000),1000)
    pred_lam_1 = y_pred_reshape[:,:5]
    gt_lam = test_labels.reshape(int(len(test_labels)/1000),1000)
    first_gt_lam = gt_lam[:,:5]
    errors = abs(pred_lam_1 - first_gt_lam)
    mape = 100 * np.mean(errors / first_gt_lam)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


base_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
base_model.fit(X_train, y_train)


base_model = RandomForestRegressor(n_estimators = 30, random_state = 42)
base_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
y_pred_reshape = y_pred.reshape(int(len(y_pred)/1000),1000)
pred_lam_1 = y_pred_reshape[:,:5]
gt_lam = y_val.reshape(int(len(y_val)/1000),1000)
first_gt_lam = gt_lam[:,:5]

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_val, y_val)
rand_y_pred = best_random.predict(X_val)
rand_y_pred_reshape = rand_y_pred.reshape(int(len(rand_y_pred)/1000),1000)
rand_pred_lam_1 = rand_y_pred_reshape[:,:5]
rand_gt_lam = y_val.reshape(int(len(y_val)/1000),1000)
first_gt_lam = rand_gt_lam[:,:5]

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

rf_random.best_params_

# Initialize an empty list to store the predictions for each lambda index
predictions_dict = {}

# Loop over the first ten lambda indices (1 to 10)
for l in range(1, 11):
    # Create a list to store predictions for the current lambda index
    lambda_predictions = []
    
    gt_lambda_vals = []
    W_vals = []
    s_vals = []
    p0_vals = []
    lam_approx_vals = []

    gt_lambda = batches_lambda[l-1]
    data_noisy = batches_noisy_data[l-1]
    W = batches_W[l-1]

    dp_lambda_approx = batches_approx[l-1]
    s_values = batches_svals[l-1]
    p0 = batches_p0[l-1]
    # Loop over the 100 noise realizations for this lambda index
    for i in range(100):
        # Generate noisy data for the current realization

        # Select the corresponding lambda value for this index

        # Call the random_forest function to predict the lambda value
        # (Note: You may need to modify the function to take 'k' as an argument)
        predictions, lams = random_forest(l, data_noisy[i], W[i], gt_lambda[i])

        # Append the predictions to the list
        lambda_predictions.append(predictions)
        gt_lambda_vals.append(lams)

        W_vals.append(W[i])
        s_vals.append(s_values[i])
        lam_approx_vals.append(dp_lambda_approx[i])
        p0_vals.append(p0[i])

    # Append the list of predictions for this lambda index to the overall list
    predictions_dict[l] = {
        'predictions': lambda_predictions,
        'ground_truth_lambda': gt_lambda_vals,  # Convert to list for easier access
        'lambda_approx': lam_approx_vals,
        'W_vals:': W_vals,
        'sing_vals': s_vals,
        'p0_vals': p0_vals
    }

file_path = 'predictions_dict.pkl'
import pickle
import scipy.io
import os
import pickle
import sys
sys.path.append('/Users/steveh/Downloads/NIH 23-24/LocReg_Python')
FileName = 'predictions_dict'
directory_path = os.getcwd()
os.makedirs(directory_path, exist_ok=True)
# Construct the full file path using os.path.join()
file_path = os.path.join(directory_path, FileName + '.pkl')
# Save the dictionary to a binary file using pickle
with open(file_path, 'wb') as file:
    pickle.dump(predictions_dict, file)
print(f"File saved at: {file_path}")



def calculate_error(dictionary):
    total_err_dict = {}
    for i in range(1,len(dictionary) + 1):
        p0_curve_err = []
        p1_curve_err = []
        err_list = []
        for j in range(100):
            values_err = np.linalg.norm(dictionary[i]['predictions'][j] - dictionary[i]['ground_truth_lambda'][j])
            err_list.append(values_err)
            p0 = dictionary[i]['p0_vals'][j]
            dictionary[i]['lambda_approx'][j][:0] = dictionary[i]['predictions'][j][:0]
            p1 = dictionary[i]['W_vals:'][j] @ (dictionary[i]['sing_vals'][j] / (dictionary[i]['sing_vals'][j] **2 + dictionary[i]['lambda_approx'][j]))
            p0_err = np.linalg.norm(g - p0)
            p1_err = np.linalg.norm(g - p1)
            p0_curve_err.append(p0_err)
            p1_curve_err.append(p1_err)
        total_err_dict[i] = {
            'avg_val_L2_error': np.mean(err_list),
            'avg_p0_l2_error': np.mean(p0_curve_err),
            'avg_p1_l2_error': np.mean(p1_curve_err)
        }
    return total_err_dict

err_dict = calculate_error(predictions_dict)

predictions_dict[2]['predictions'][0]
predictions_dict[2]['ground_truth_lambda'][0]
err_dict[10]['avg_p1_l2_error']
err_dict[10]['avg_val_L2_error']
err_dict[10]['avg_p0_l2_error']

p1_avg_errors = []
p0_avg_errors = []
# Iterate over the range from 1 to 10 (inclusive)
for i in range(1, 11):
    avg_error = err_dict[i]['avg_p1_l2_error']
    p1_avg_errors.append(avg_error)
    p0_avg_error = err_dict[i]['avg_p0_l2_error']
    p0_avg_errors.append(p0_avg_error)

# Calculate the average of 'avg_p1_l2_error' values
median_p1 = np.median(p1_avg_errors) 
median_p0 = np.median(p0_avg_errors)

# Print the average
print("median p1:", median_p1)
print("median p0:", median_p0)










#data_vector contains all 1000 NR
#data_source is the subset
def rand_forest_simul(lambda_index, batched_data, batched_lams, batched_W):
    dat_subset = np.array(batched_data[lambda_index])
    lam_subset = np.array(batched_lams[lambda_index])
    Ws_subset = np.array(batched_W[lambda_index])
    k = lambda_index 
    dat_source = dat_subset
    W = Ws_subset
    gt_lambdas = lam_subset
    preds_batch = random_forest(k, dat_source, W, gt_lambdas)
    print(f'This is the result for {len(dat_subset)} NR for predicting the lambdas in position {lambda_index}')
    
    return preds_batch, lam_subset


predicted, actual = rand_forest_simul(lambda_index = 1, batched_data = batches_noisy_data, batched_lams= batches_lambda, batched_W= batches_W)

# lambda_index = 1
# batched_data = batches_noisy_data
# batched_lams= batches_lambda
# batched_W= batches_W

# s_subset = p0[:k]

#Problem desciption
#10000 NR; 10 l.c for all these; find coefficient alphas; wihtout ground truth; simulations based on ; simulations on each gaussian; 
#for each unit vector for len(g), train 1000 NR ; so total of 10000NR for 10 g, len(Dataset 10000) (10000,10).

#Only predict the first value of lambda; so vector (10000,1); discretization could be 20; set fixed SNR to 1000; 1000 NR, (10000 instance, 20)

#1 unit vector traiing on basis and predict on l.c of basis

#Try W if 

#Predict 1st lambda value, if work then next 5 lambdas, then 10 lambdas

#

#Monte-Carlo

#Iterative; if we know 1st lambda, can you predict 2nd lambda,; most V.T solution space; each column perpendicul ar to other. natural basis for G
#First lambda add up maginutd eof first column,



plt.plot(lambda_result[0:10], label='gt')
plt.plot(y_pred[0:10], label='random forest')
plt.legend()
plt.show()

lambda_approx_orig = lambda_approx
# lambda_approx = lambda_approx_orig
# lambda_approx[:10] = lambda_result[:10]

lambda_approx[:10] = predictions[:10]
# lambda_approx_test[:2] = lambda_mod_result[:2]

# Calculate p1
p1 = W @ (s / (s**2 + lambda_approx))

np.linalg.norm(g-p0)
np.linalg.norm(g-p1)



plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(p0, label='p0')
plt.plot(p1, label='p1')
# plt.plot(p1_nn, label='p1_nn')
plt.plot(g, label='GT')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(lambda_result, label='lambda GT')
plt.plot(lambda_approx, label='lambda approx')
# plt.plot(lambda_approx_test, label='lambda approx_nn')

plt.legend()
plt.show()



# # Train a Regression Model:
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras import layers

# # Prepare the training data
# X = np.hstack((data_subset.reshape(-1,1), W_subset))

# # Reshape X to match the expected shape of (samples, features)
# X = np.reshape(X, (X.shape[0], -1))

# y = lambda_result[:10]  # Replace 'lambda_result_subset' with the corresponding subset of lambda_result

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Normalize the input features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # Define the neural network model
# model = tf.keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     layers.Dense(32, activation='relu'),
#     layers.Dense(1)  # Output layer with a single node for regression
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# new_data_subset = data_noisy[k-1:-1]
# new_W_subset =W[k-1:-1,:]
# # Use the trained model for prediction
# new_X = np.hstack((new_data_subset.reshape(-1,1), new_W_subset))  # Replace 'new_data_subset' and 'new_W_subset' with your new data
# new_X = np.reshape(new_X, (new_X.shape[0], -1))
# X_new_scaled = scaler.transform(new_X)
# predictions = model.predict(X_new_scaled)


]# Choose a regression model suitable for your data, such as linear regression, decision trees, random forests, or neural networks.
# Use the training dataset to train the regression model. The input features should be the selected subset of data_noisy and W, and the target variable should be the corresponding subset of lambda_result.
# Predict Lambda Values:

# Once the regression model is trained, use it to predict the values of lambda_result for the remaining data points or any new data.
# Provide the necessary input features (e.g., corresponding subset of data_noisy and W) to the trained regression model, and obtain the predicted values of lambda_result.