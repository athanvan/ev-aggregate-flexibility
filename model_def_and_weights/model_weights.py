import torch
import numpy as np
import cvxpy as cp

def strip_model_data_new(model):
    # Get all ICNN weights
    fc_weights = []
    for layer in model.fc_layers:
        fc_weights.append(layer.weight)
    rl_weights = []
    rl_biases = []
    for layer in model.rl_layers:
        rl_weights.append(layer.weight)
        rl_biases.append(layer.bias)
    pmin, pmax = model.bcl.pmin, model.bcl.pmax
    return fc_weights, rl_weights, rl_biases, pmin, pmax


def construct_problem_data_cvxpy(n, fc_weights, rl_weights, rl_biases, pmin, pmax):
    """
    Creates cvxpy constant matrices/vectors C, d matching Appendix A
    """
    num_zs_all = sum([layer.shape[0] for layer in rl_weights])
    
    # construct the constraint matrix C
    Ds = cp.vstack(rl_weights)
    Es = [cp.hstack([cp.Constant(-np.eye(rl_weights[0].shape[0])),
                     cp.Constant(np.zeros((rl_weights[0].shape[0], num_zs_all - rl_weights[0].shape[0])) if num_zs_all - rl_weights[0].shape[0] > 0 else np.zeros((rl_weights[0].shape[0], 1)))])]
    for i in range(1, len(rl_weights)):
        new_E = []
        row_len = 0
        if i > 1:
            zero_block_size = sum([fc_weights[j].shape[1] for j in range(i-1)])
            if zero_block_size > 0:
              new_E.append(
                  cp.Constant(np.zeros((rl_weights[i].shape[0], zero_block_size))))
              row_len += zero_block_size
        new_E.append(fc_weights[i-1])
        row_len += fc_weights[i-1].shape[1]
        new_E.append(cp.Constant(-np.eye(rl_weights[i].shape[0])))
        row_len += rl_weights[i].shape[0]
        zero_block_size = num_zs_all - row_len
        if zero_block_size > 0:
          new_E.append(cp.Constant(np.zeros((rl_weights[i].shape[0], zero_block_size))))
        Es.append(cp.hstack(new_E))
        
    # stack them together
    C = cp.hstack([Ds, cp.vstack(Es)])
    
    # add z constraints
    B, x_zeros = -np.eye(C.shape[1] - n), np.zeros((C.shape[1]-n,n))
    # this deals with constraint 7c / third to last row of C in Appendix A
    B[-1][-1] = -1 * B[-1][-1]
    B = cp.hstack([cp.Constant(x_zeros), cp.Constant(B)])
    C = cp.vstack([C, B])
    
    # add pmin and pmax bounds
    col_C = C.shape[1]
    zero_block = cp.hstack([cp.Constant(np.eye(n)), cp.Constant(np.zeros((n, col_C - n)))])
    C = cp.vstack([C, zero_block, -zero_block])

    # make d vec
    d = -cp.hstack(rl_biases)
    d = cp.hstack([d, cp.Constant(np.zeros((C.shape[1] - n)))])
    d = cp.hstack([d, pmax,-pmin])
    return C, d

def construct_problem_data_fixed(model, device):
    """
    Creates numpy constant matrices/vectors C, d matching Appendix A
    """
    n = model.input_dim
    fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(model)
    fc_weights = [p.data.cpu().numpy() for p in fc_weights]
    rl_weights = [p.data.cpu().numpy() for p in rl_weights]
    rl_biases = [p.data.cpu().numpy() for p in rl_biases]
    pmin = pmin.detach().cpu().numpy()
    pmax = pmax.detach().cpu().numpy()


    num_zs_all = sum([layer.shape[0] for layer in rl_weights])
   
    # construct the constraint matrix C
    Ds = np.vstack(rl_weights)
    Es = [np.hstack([-np.eye(rl_weights[0].shape[0]), np.zeros((rl_weights[0].shape[0], num_zs_all - rl_weights[0].shape[0]))])]
    for i in range(1, len(rl_weights)):
        new_E = []
        row_len = 0
        if i > 1:
            new_E.append(np.zeros((rl_weights[i].shape[0], sum(
                [fc_weights[j].shape[1] for j in range(i-1)]))))
            row_len += sum([fc_weights[j].shape[1] for j in range(i-1)])
        new_E.append(fc_weights[i-1])
        row_len += fc_weights[i-1].shape[1]
        new_E.append(-np.eye(rl_weights[i].shape[0]))
        row_len += rl_weights[i].shape[0]
        new_E.append(np.zeros((rl_weights[i].shape[0], num_zs_all - row_len)))
        Es.append(np.hstack(new_E))
        
    # stack them together
    C = np.hstack([Ds, np.vstack(Es)])
    
    # add z constraints
    B, x_zeros = -np.eye(C.shape[1] - n), np.zeros((C.shape[1]-n,n))
    # this deals with constraint 7c / third to last row of C in Appendix A
    B[-1][-1] = -1 * B[-1][-1]
    B = np.hstack([x_zeros, B])
    C = np.vstack([C, B])
    
    # add pmin and pmax bounds
    col_C = C.shape[1]
    zero_block = np.hstack([np.eye(n), np.zeros((n, col_C - n))])
    C = np.vstack([C, zero_block, -zero_block])
    
    # make d vec
    d = -np.hstack(rl_biases)
    d = np.hstack([d, pmax, -pmin])
    d = np.hstack([d, np.zeros((C.shape[1] - n))])
    return C, d

def create_A_matrix(timesteps, col_C):
  A = np.hstack([np.eye(timesteps), np.zeros((timesteps, col_C - timesteps))])
  return A