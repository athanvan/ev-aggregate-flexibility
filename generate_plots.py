from plotting.plotting_method import plot_icnn_slices
import torch
from data_generation.create_load_data import load_household_15min, build_load_profiles, f_reshape, f_vec, blkdiag_repeat
import numpy as np 
from data_generation.create_flexibility_sets import calculate_indiv_sets, find_chebyshev_center
from model_def_and_weights.taha_models import general_affine_inner_approx, struct_preserve_inner_approx
from model_def_and_weights.icnn_definition import ICNN
from comparison_methods.comparison import optimal_ppm

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

seed = 4

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
h_ga = h_ga + H_ga @ (-1 * summed_center)

alpha, pbar_sp = struct_preserve_inner_approx(H, h_i, hx, N)
h_sp = alpha * hx + H @ pbar_sp
H_sp = H
h_sp = h_sp + H_sp @ (-1 * summed_center)

H_block = blkdiag_repeat(H, N)
h_full = f_vec(h_i)

h_i_translated = [h_i[:, i] - H @ cheb_centers[i] for i in range(N)]

icnn_model = ICNN(T, T*4, 1, 1, torch.as_tensor([-500] * T), torch.as_tensor([500] * T), H_ga, h_ga)
checkpoint = torch.load(f'model_checkpoints/ppm_model_checkpoint_{seed}.pth') 
icnn_model.load_state_dict(checkpoint['model_state_dict'])

opt_solutions = torch.zeros(base_loads.shape)
for m in range(base_loads.shape[1]):
    new_soln_opt = optimal_ppm(T, N, base_loads[:, m], H_block, h_full, return_u = True)
    opt_solutions[:, m] = torch.as_tensor(new_soln_opt - summed_center)

figs = plot_icnn_slices(
                    opt_solutions[:, 0],
                    icnn_model, 
                    H,
                    h_i_translated,
                    ga_model = (H_ga, h_ga),
                    sp_model = (H_sp, h_sp)
                )

for idx, figure in enumerate(figs):
    figure.savefig(f"plots/figure_{idx}")