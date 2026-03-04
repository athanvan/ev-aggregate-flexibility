import numpy as np
import torch 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model_def_and_weights.icnn_definition import ICNN
from model_def_and_weights.taha_models import general_affine_inner_approx, struct_preserve_inner_approx
from data_generation.create_flexibility_sets import calculate_indiv_sets, find_chebyshev_center
from data_generation.create_load_data import load_household_15min, build_load_profiles, f_reshape, f_vec, blkdiag_repeat
from comparison_methods.comparison import taha_model_ppm, icnn_ppm, optimal_ppm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

T = 18
N = 25
delta = 1

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

load_train, load_other = train_test_split(base_loads.T, test_size=0.4, random_state=42)
load_val, load_test = train_test_split(load_other, test_size=0.5, random_state=42)
load_train, load_val, load_test = load_train.T, load_val.T, load_test.T

ml_minus_opt = [] 
ga_minus_opt = []
sp_minus_opt = []
ml_mean = []
ga_mean = []
sp_mean = []

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
    
    hx = np.sum(h_i, axis=1) / N  
    P, pbar_ga = general_affine_inner_approx(H, h_i, hx, N)
    h_ga = hx + H @ np.linalg.inv(P) @ pbar_ga
    H_ga = H @ np.linalg.inv(P)
    
    alpha, pbar_sp = struct_preserve_inner_approx(H, h_i, hx, N)
    h_sp = alpha * hx + H @ pbar_sp
    H_sp = H
    
    H_block = blkdiag_repeat(H, N)
    h_full = f_vec(h_i)
    
    icnn_model = ICNN(T, T*4, 1, 1, torch.as_tensor([-500] * T), torch.as_tensor([500] * T), H_ga, h_ga)
    checkpoint = torch.load(f'model_checkpoints/ppm_model_checkpoint_{seed}.pth') 
    icnn_model.load_state_dict(checkpoint['model_state_dict'])
    
    curr_ml, curr_ga, curr_sp = 0, 0, 0
    for t in range(load_test.shape[1]):
        l = load_test[:, t]
        opt_l = optimal_ppm(T, N, l, H_block, h_full)
        
        ml_result = icnn_ppm(T, N, l, summed_center, icnn_model)
        curr_ml += ml_result - opt_l
        ml_minus_opt.append(ml_result - opt_l)
        
        ga_result = taha_model_ppm(T, l, H_ga, h_ga)
        curr_ga += ga_result - opt_l
        ga_minus_opt.append(ga_result - opt_l)
        
        sp_result = taha_model_ppm(T, l, H_sp, h_sp)
        curr_sp += sp_result - opt_l
        sp_minus_opt.append(sp_result - opt_l)
    
    ml_mean.append(curr_ml/load_test.shape[1])
    ga_mean.append(curr_ga/load_test.shape[1])
    sp_mean.append(curr_sp/load_test.shape[1])
    
# make boxplot
gaps = np.zeros((load_test.shape[1] * 10, 3))
gaps[:, 0] = np.array(ml_minus_opt)
gaps[:, 1] = np.array(sp_minus_opt)
gaps[:, 2] = np.array(ga_minus_opt)
plt.figure()
plt.boxplot(gaps, notch=False)
plt.ylabel("Peak power gap (kW)")
plt.xticks([1, 2, 3], ["ICNN", "struc.-pres.", "gen. affine"])
plt.title("Optimality Gap on Peak Power Minimization")
plt.savefig("boxplot_result.png")

print(f"Mean ICNN Gap: {np.mean(ml_mean)}")
print(f"ICNN Standard Error: {np.std(ml_mean)/(9 ** 0.5)}")
print(f"Mean General Affine Gap: {np.mean(ga_mean)}")
print(f"General Affine Standard Error: {np.std(ga_mean)/(9 ** 0.5)}")
print(f"Mean Structure Preserving Gap: {np.mean(sp_mean)}")
print(f"Structure Preserving Standard Error: {np.std(sp_mean)/(9 ** 0.5)}")
    
    
 