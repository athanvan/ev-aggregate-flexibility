from model_def_and_weights.icnn_definition import ICNN
from containment_problem.containment import CVXPY_Containment_Problem
from training_methods.ppm_epoch import PPM_Epoch
from training_methods.ppm_loss import PPM_Loss
from comparison_methods.comparison import icnn_ppm, optimal_ppm
import numpy as np
import os
import shutil
import torch.optim as optim
from torch import nn
import torch
from tqdm import tqdm
import time 
import psutil
import wandb

def train_icnn(H_i, h_i, pmin, pmax, device, info, params_dict):
    # Unpack parameters
    lr = params_dict['lr']
    epochs = params_dict['epochs']
    hidden_width = params_dict['hidden_width']
    hidden_depth = params_dict['hidden_depth']
    chosen_solver = params_dict['solver']
    verbose = params_dict['verbose']
    H_init = params_dict["H_init"]
    h_init = params_dict["h_init"]
    seed = params_dict["seed"]
    run_name = params_dict["run_name"]
    H_block = params_dict["H_block"]
    h_full = params_dict["h_full"]
    
    # setting seed for reproducibility
    torch.manual_seed(4327)
    torch.cuda.manual_seed(4327)
    T = H_i[0].shape[1]
    N = len(H_i)

    model = ICNN(T, hidden_width, 1, hidden_depth,
                pmin, pmax, H_init, h_init, 
                sigmoid_output=False).to(device)
  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    containment_problem = CVXPY_Containment_Problem(model, T, N, H_i, h_i,
                                                    chosen_solver = chosen_solver, 
                                                    verbose = verbose)
    task_loss_problem = PPM_Loss(info["translation"], 1, T, N, 
                                 model, info["train_batch_size"], 
                                 info["val_batch_size"], verbose)
        
    epoch_methods = PPM_Epoch(info["load_train"], info["load_val"],
                              info["translation"], model, optimizer, task_loss_problem, 
                              containment_problem, device, N, T)
                              
    pbar = tqdm(total=epochs,
                desc='Training', dynamic_ncols=True)
    
    train_loss = []
    val_loss = []
    ratio = []
    
    optimal_ppm_sols = []
    for t in range(info["load_val"].shape[1]):
        l = info["load_val"][:, t]
        optimal_ppm_sols.append(optimal_ppm(T, N, l, H_block, h_full))
    
    wandb.init(project="Train ICNN for PPM",
            name = run_name)
    
    for _ in range(epochs):
        return_vals = epoch_methods.ppm_epoch(seed)
        ratio_epoch = return_vals["ratio"]
        train_loss_epoch = return_vals["ppm_train_loss"]
        val_loss_epoch = return_vals["ppm_val_loss"]
        
        ml_minus_opt = []
        for t in range(info["load_val"].shape[1]):
            l = info["load_val"][:, t]
            ml_minus_opt.append(icnn_ppm(T, N, l, info["translation"], model) - optimal_ppm_sols[t])
            
        wandb.log({"ratio": ratio_epoch, "train loss": train_loss_epoch,
                   "val loss": val_loss_epoch, "optimality gap": np.mean(ml_minus_opt)})
        
        ratio.append(ratio_epoch)
        val_loss.append(val_loss_epoch)
        train_loss.append(train_loss_epoch)
        
        pbar.set_postfix_str(
        f"Train Loss={train_loss_epoch:.4f} | Val Loss={val_loss_epoch:.4f} | Containment={float(ratio_epoch):.4f} ")

        pbar.update(1)
        
        if np.mean(ml_minus_opt) < 0.001:
            break
        
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        ratio, status = containment_problem.solve_icnn()
        assert status == "optimal"
        model.pgd(ratio)
    
    wandb.finish()
       
    return model, train_loss, val_loss, ratio
