import torch
import cvxpy as cp 
import numpy as np
from training_methods.ppm_loss import PPM_Loss

def taha_model_ppm(T, l, H, h):
    # compute value for peak power minimization for Taha (eq 24)
    u = cp.Variable(T)
    constraints = [H @ u <= h]
    
    objective = cp.Minimize(cp.norm_inf(u + l))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CUOPT", verbose=False)
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("status:", prob.status)
        
    return prob.value

def optimal_ppm(T, N, l, H_block, h_full, return_u = False):
    # compute optimal value for peak power minimization
    zi = cp.Variable((T, N))
    u = cp.Variable(T, name = "u")
    
    constraints = [cp.sum(zi, axis=1) == u]
    constraints += [H_block @ cp.vec(zi) <= h_full]
    
    objective = cp.Minimize(cp.norm_inf(u + l))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="CUOPT", verbose=False)
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("optimal status:", prob.status)
    
    if return_u:
        return prob.var_dict["u"].value
        
    return prob.value

def icnn_ppm(T, N, l, translation, model):
    # use ppm loss class to evaluate peak power minimization value for ICNN
    ppm_loss = PPM_Loss(translation, 1, T, N, model, 1, 1, False)
    return ppm_loss.ppm_evaluate(torch.as_tensor(l).unsqueeze(1), layer = False)