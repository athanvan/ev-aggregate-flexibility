import numpy as np
import torch
import wandb
import time 

class PPM_Epoch():
    def __init__(self, load_train, load_val, translation,
                 model, optimizer, ppm_loss, 
                 containment_problem, device, 
                 N, T):
        """
        Args: 
            load_train: load data training set
            load_val: load data validation set
            translation: shift of aggregate set U (∆) 
            optimal_sol: optimal charging schedule for peak power
                          minimization problem for some load vector
            model: ICNN
            optimizer: optimizer 
            ppm_loss: loss function for peak power minimization problem
            containment_problem: class to find scaling ratio
            comparison_models: Taha models to compare ICNN against 
            device: what device to move tensors to 
            N: number of EVs 
            T: number of timesteps
        """
        self.load_train, self.load_val = load_train, load_val
        self.translation = translation
        self.model = model 
        self.optimizer = optimizer 
        self.ppm_loss = ppm_loss
        self.contain_class = containment_problem
        self.device = device
        self.best_val_loss = None
        self.N, self.T = N, T 

    def ppm_epoch(self, seed):
        """
        - Runs an epoch of training with the peak power minimization loss function
        - Computes and saves training and validation loss 
        """        
        return_vals = {}
        
        ratio = 1
        with torch.no_grad():
            ratio, status = self.contain_class.solve_icnn()
            assert status == "optimal"
            self.model.pgd(ratio)
            return_vals["ratio"] = ratio
        load_val = torch.as_tensor(self.load_val).to(self.device).float()
        ppm_val_loss = self.ppm_loss.ppm_evaluate(load_val, layer = False)
        return_vals["ppm_val_loss"] = ppm_val_loss
        
        # save model if it's current best 
        if self.best_val_loss is None or ppm_val_loss < self.best_val_loss:
            self.best_val_loss = ppm_val_loss
            torch.save({'model_state_dict': self.model.state_dict()}, 
                        f'model_checkpoints/ppm_model_checkpoint_{seed}.pth')
                
        self.optimizer.zero_grad()
        load_train = torch.as_tensor(self.load_train).to(self.device).float()
        translation = torch.tensor(self.translation).to(self.device)  
        sol = self.ppm_loss.ppm_evaluate(load_train, layer = True)
        # recompute task loss with the solution from cvxpylayer 
        ppm_training_loss = 0
        for idx in range(load_train.shape[1]):
            ppm_training_loss += torch.linalg.norm(sol[0][:, idx] + translation + load_train[:, idx], ord = float("inf"))
        ppm_training_loss /= self.load_train.shape[1]
        
        ppm_training_loss.backward()  
        self.optimizer.step()
        # to preserve convexity
        self.model.clamp_weights()
        
        # save loss value 
        return_vals["ppm_train_loss"] = torch.clone(ppm_training_loss).detach().cpu().numpy()
    
        return return_vals
        
