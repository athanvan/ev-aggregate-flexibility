import torch
import cvxpy as cp 
import numpy as np
from training_methods.ppm_loss import PPM_Loss

def taha_model_ppm(T, l, H, h, return_u = False, solver = "CUOPT"):
    # compute value for peak power minimization for Taha (eq 24)
    u = cp.Variable(T, name = "u")
    constraints = [H @ u <= h]
    
    objective = cp.Minimize(cp.norm_inf(u + l))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("status:", prob.status)
    
    if return_u:
        return prob.var_dict["u"].value
    
    return prob.value

def optimal_ppm(T, N, l, H_block, h_full, return_u = False, solver = "CUOPT"):
    # compute optimal value for peak power minimization
    zi = cp.Variable((T, N))
    u = cp.Variable(T, name = "u")
    
    constraints = [cp.sum(zi, axis=1) == u]
    constraints += [H_block @ cp.vec(zi) <= h_full]
    
    objective = cp.Minimize(cp.norm_inf(u + l))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=False)
    
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print("optimal status:", prob.status)
    
    if return_u:
        return prob.var_dict["u"].value
        
    return prob.value

def icnn_ppm(T, N, l, translation, model, return_u = False, solver = "CUOPT"):
    # use ppm loss class to evaluate peak power minimization value for ICNN
    ppm_loss = PPM_Loss(translation, 1, T, N, model, 1, 1, False, solver)
    return ppm_loss.ppm_evaluate(torch.as_tensor(l).unsqueeze(1), layer = False, return_u=return_u)