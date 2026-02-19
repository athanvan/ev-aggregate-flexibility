# %%
import torch
from data_generation.create_load_data import load_household_15min, build_load_profiles, f_reshape, f_vec, blkdiag_repeat
import numpy as np 
from data_generation.create_flexibility_sets import calculate_indiv_sets, find_chebyshev_center
from model_def_and_weights.taha_models import general_affine_inner_approx
from model_def_and_weights.icnn_definition import ICNN
from comparison_methods.comparison import optimal_ppm, taha_model_ppm, icnn_ppm
from plotting.plotting_slice import plot_specific_slice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

T = 18
N = 25
delta = 1
seed = 4

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
translated_h_ga = h_ga + H_ga @ (-1 * summed_center)

H_block = blkdiag_repeat(H, N)
h_full = f_vec(h_i)

icnn_model = ICNN(T, T*4, 1, 1, torch.as_tensor([-500] * T),torch.as_tensor([500] * T), H_ga, h_ga)
checkpoint = torch.load(f'model_checkpoints/ppm_model_checkpoint_{seed}') 
icnn_model.load_state_dict(checkpoint['model_state_dict'])

idx = 0
u0 = optimal_ppm(T, N, base_loads[:, idx], H_block, h_full, return_u = True)
u_icnn = icnn_ppm(T, N, base_loads[:, idx], summed_center, icnn_model, return_u = True)[:, 0] + summed_center
u_taha = taha_model_ppm(T, base_loads[:, idx], H_ga, h_ga, return_u = True) 

h_i_translated = [h_i[:, i] - H @ cheb_centers[i] for i in range(N)]
fig = plot_specific_slice(
                    u0 - summed_center,
                    u_icnn - summed_center,
                    u_taha - summed_center, 
                    icnn_model, 
                    H,
                    h_i_translated,
                    ga_model = (H_ga, translated_h_ga)
                )
fig.savefig("slice.pdf", pad_inches=0, bbox_inches='tight')
