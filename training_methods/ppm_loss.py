import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from model_def_and_weights.model_weights import strip_model_data_new
import torch
import numpy as np
import wandb

class PPM_Loss:
    def __init__(self, translation,
                 delta, T, N, model, train_batch_size, 
                 val_batch_size, if_verbose):
        """
        Args: 
            translation: shift of aggregate set U (∆) 
            delta: length of timestep (1)
            T: number of timesteps
            N: number of vehicles
            model: ICNN
            batch_size: number of load profiles
            if_verbose: Show log for solving cvxpy problem
        """
        self.translation = translation 
        self.delta = delta
        self.T = T
        self.N = N
        self.model = model
        
        self.ppm_init(train_batch_size, True)
        self.ppm_init(val_batch_size, False)
        
        self.keywords = dict(
          verbose = if_verbose
        )
    
    def ppm_init(self, batch_size, if_layer):
        """
        Create charging schedules u and parameters for peak power minimization problem 
        """
        u = cp.Variable((self.T, batch_size))
        var_list  = [u]
        
        l = cp.Parameter((self.T, batch_size))
        param_list = [l]
        
        self.create_ppm_constraints(u, l, param_list, var_list, 
                                    if_layer, batch_size)

       
    def create_ppm_constraints(self, u, l, param_list, var_list, 
                               if_layer, batch_size):
        """
        - Create constraints and objectives for peak power minimization problem.
        - Create this problem as a cvxpylayer (to allow backpropagation) if if_layer
          is True or as a cvxpy problem if if_layer is False. 
        
            u: Variable for charging schedules 
            u_scaled: Variable for charging schedule multipled by scaling ratio
            l: Parameter for load profiles
            param_list: List of parameters for problem
            var_list: List of variables for problem
            if_layer: Cvxpylayer created if true, cvxpy problem created if false
            batch_size: Number of load profiles
        """
        fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(self.model)
        pmin_param = cp.Parameter(pmin.shape)
        pmax_param = cp.Parameter(pmax.shape)
        param_list.append(pmin_param)
        param_list.append(pmax_param)
        
        fcw_params = []
        rlw_params = []
        rlb_params = []
        
        for i in range(len(fc_weights)):
            fcw = cp.Parameter(fc_weights[i].shape)
            param_list.append(fcw)
            fcw_params.append(fcw)
            
        for i in range(len(rl_weights)):
            rlw = cp.Parameter(rl_weights[i].shape)
            param_list.append(rlw)
            rlw_params.append(rlw)
        
        for i in range(len(rl_biases)):
            rlb = cp.Parameter(rl_biases[i].shape)
            param_list.append(rlb)
            rlb_params.append(rlb)
        
        zs = []
        for i in range(len(rl_weights)):
            curr_z = cp.Variable((rl_weights[i].shape[0], u.shape[1]), name = f"z_{i}")
            var_list.append(curr_z)
            zs.append(curr_z)
                    
        # ICNN constraints to ensure chosen charging schedules
        constraints = [zs[0] >= 0,
                   zs[0] >= rlw_params[0] @ u + cp.reshape(rlb_params[0], (rlb_params[0].shape[0], 1))]
        for i in range(1, len(rl_weights)-1):
            constraints.append(zs[i] >= 0)
            constraints.append(zs[i] >= fcw_params[i-1] @ zs[i-1] +
                            rlw_params[i] @ u + cp.reshape(rlb_params[i], (rlb_params[i].shape[0], 1)))
        constraints.append(u <= cp.reshape(pmax_param, (pmax_param.shape[0], 1)))
        constraints.append(u >= cp.reshape(pmin_param, (pmin_param.shape[0], 1)))
        constraints.append(zs[-1] == fcw_params[-1] @
                        zs[-2] + rlw_params[-1] @ u + cp.reshape(rlb_params[-1], (rlb_params[-1].shape[0], 1)))
        constraints.append(zs[-1] <= 0)
        
        obj = 0
        for idx in range(batch_size):
            obj += cp.norm_inf(u[:, idx] + cp.Constant(self.translation) + l[:, idx])
            
        if if_layer: 
            self.ppm_problem_train = cp.Problem(cp.Minimize(obj/batch_size), constraints)
            self.ppm_layer = CvxpyLayer(self.ppm_problem_train, parameters = param_list, variables = var_list)
        else:
            self.ppm_problem_val = cp.Problem(cp.Minimize(obj/batch_size), constraints)
            self.ppm_val_params = param_list 
        
        
    def ppm_evaluate(self, l, layer = True):
        """
        For load profiles, find the value of the peak power minimization problem.
        
            l: load profiles to evaluate the problem on
            layer: whether or not to evaluate using cvxpylayer
        """
        fc_weights, rl_weights, rl_biases, pmin, pmax = strip_model_data_new(self.model)
        if layer: 
            try:
                sol = self.ppm_layer(
                    *([l, pmin, pmax] + fc_weights + rl_weights + rl_biases), solver_args = self.keywords)
            except Exception as e:
                raise e
            return sol
        
        else:
            idx = 0
            self.ppm_val_params[idx].value = l.cpu().detach().numpy()
            idx += 1
            self.ppm_val_params[idx].value = pmin.cpu().detach().numpy()
            idx += 1 
            self.ppm_val_params[idx].value = pmax.cpu().detach().numpy()
            idx += 1 
            for fc_w in fc_weights:
                self.ppm_val_params[idx].value = fc_w.cpu().detach().numpy()
                idx += 1
            for rl_w in rl_weights:
                self.ppm_val_params[idx].value = rl_w.cpu().detach().numpy()
                idx += 1
            for rl_b in rl_biases:
                self.ppm_val_params[idx].value = rl_b.cpu().detach().numpy()
                idx += 1
            self.ppm_problem_val.solve(solver = "CUOPT", **self.keywords)
            return self.ppm_problem_val.value