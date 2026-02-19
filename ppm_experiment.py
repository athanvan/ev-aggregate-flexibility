import numpy as np
import torch 
from sklearn.model_selection import train_test_split

from model_def_and_weights.taha_models import general_affine_inner_approx, struct_preserve_inner_approx
from training_methods.train_loop import train_icnn
from data_generation.create_flexibility_sets import calculate_indiv_sets, find_chebyshev_center
from data_generation.create_load_data import load_household_15min, build_load_profiles, f_reshape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 18
N = 25
delta = 1

icnn_solutions = []
general_affine_solutions = []
struct_preserve_solutions = []
optimal_solution = []

for seed in range(10):
    a = np.ones(N)
    d = T * np.ones(N)
    L = delta * np.tril(np.ones((T, T)))
    H = np.vstack([L, -L, np.eye(T), -np.eye(T)])

    h_i = calculate_indiv_sets(a, d, N, T, seed)  
    cheb_centers = []
    for i in range(N):
        problem = find_chebyshev_center(H, h_i[:, i])
        cheb_centers.append(problem.var_dict["center"].value) 

    summed_center = sum(cheb_centers) 
    
    # initialize to eq. 27 from Taha et al 2024
    hx = np.sum(h_i, axis=1) / N  
    P, pbar_ga = general_affine_inner_approx(H, h_i, hx, N)
    h_taha = hx + H @ np.linalg.inv(P) @ pbar_ga
    H_taha = H @ np.linalg.inv(P)
    
    # translate the sets to contain the origin
    h_taha = h_taha + H_taha @ (-1 * summed_center)
    h_i_translated = [h_i[:, i] - H @ cheb_centers[i] for i in range(N)]
    H_i = [H for _ in range(N)]
    
    # read load data
    ny_15min_csv = "load_data/15minute_data_newyork.csv"

    nys_load_df = load_household_15min(ny_15min_csv)
    agg_load = build_load_profiles(nys_load_df, T_15=24*4, expected_days=184)  

    offset = 15

    # compute hourly aggregate base loads
    tmp = f_reshape(agg_load, (4, -1, agg_load.shape[1]))  
    avg_agg_load = tmp.mean(axis=0)
    base_loads_flat = avg_agg_load.ravel(order="F")

    start = offset
    end = len(base_loads_flat) - (24 - offset)
    base_loads_crop = f_reshape(base_loads_flat[start:end], (24, -1))
    base_loads = base_loads_crop[:T, :] 
    print("made base loads")

    load_train, load_other = train_test_split(base_loads.T, test_size=0.4, random_state=42)
    load_val, load_test = train_test_split(load_other, test_size=0.5, random_state=42)
    load_train, load_val, load_test = load_train.T, load_val.T, load_test.T
    augmented_loads = load_train
    while augmented_loads.shape[1] < 500:
        noise = np.random.normal(0, 0.1, size=load_train.shape) * np.mean(load_train, axis = 1).reshape((T, 1))
        augmented_loads = np.concatenate((augmented_loads, load_train + noise), axis = 1)

    # train model 
    info = {"load_train": augmented_loads, "load_val": load_val, 
            "translation": torch.as_tensor(summed_center).float().numpy(), 
             "train_batch_size": augmented_loads.shape[1], "val_batch_size": load_val.shape[1]}
    epochs = 50
    
    params = {'lr': 1e-3,                  # learning rate
        'epochs': epochs,                  # number of epochs to train the model 
        'hidden_width': T * 4,                # width of the hidden layers
        'hidden_depth': 1,                 # depth of the hidden layers
        'solver': "CUOPT", 
        'verbose': False, 
        'H_init': H_taha, 
        'h_init': h_taha,
        'seed': seed
        }
    
    model, train_loss, val_loss, ratio = train_icnn(H_i, h_i_translated, 
                        torch.as_tensor([-500] * T), 
                        torch.as_tensor([500] * T),
                        device, info,params)
       
    
    


