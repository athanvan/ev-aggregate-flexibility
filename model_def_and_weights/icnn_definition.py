import torch
import torch.nn as nn
import torch.nn.functional as F
from model_def_and_weights.bcl import BoxConstraintLayer

class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden, 
                 pmin, pmax, A, b, 
                 sigmoid_output=False):
        super().__init__()
        self.T = input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.sigmoid_output = sigmoid_output
        
        self.fc_layers = nn.ModuleList([nn.Linear(A.shape[0], A.shape[0], bias=False) for _ in range(num_hidden-1)] +
                                        [nn.Linear(A.shape[0], output_dim, bias=False)])
        self.rl_layers = nn.ModuleList([nn.Linear(input_dim, A.shape[0]) for _ in range(num_hidden)] +
                                        [nn.Linear(input_dim, output_dim)])
            
        with torch.no_grad():
            for layer in self.rl_layers:
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
                
            self.rl_layers[0].weight.copy_(torch.as_tensor(A, dtype=self.fc_layers[0].weight.dtype))
            self.rl_layers[0].bias.copy_(torch.as_tensor(-b, dtype=self.fc_layers[0].weight.dtype))
            # just training the bias from the first layer 
            self.rl_layers[0].bias.requires_grad = True
            
            I = torch.eye(self.hidden_dim, dtype=self.fc_layers[0].weight.dtype)
            for fc in self.fc_layers[:-1]:
                fc.weight.copy_(I)
                fc.weight.requires_grad = False
                
            self.clamp_weights()
            
            ones_last = torch.ones(self.output_dim, A.shape[0],
                            dtype = self.fc_layers[-1].weight.dtype)
            self.fc_layers[-1].weight.copy_(ones_last)
            self.fc_layers[-1].weight.requires_grad = False
            
        self.bcl = BoxConstraintLayer(pmin, pmax, c=1)

    def forward(self, x):
        x1 = F.relu(self.rl_layers[0](x))
        for fc_layer, rl_layer in zip(self.fc_layers[:-1], self.rl_layers[1:-1]):
            x1 = F.relu(fc_layer(x1) + rl_layer(x))
        x1 = self.fc_layers[-1](x1) + self.rl_layers[-1](x) 
        x1 = torch.maximum(x1, self.bcl(x).unsqueeze(1))
        if self.sigmoid_output:
            x1 = torch.sigmoid(x1)
        return x1

    def clamp_weights(self):
        """
        Clamps all weights in fc_layers to be >= 0 (for convexity)
        and final rl_layer to be = 0 (to ensure output sublevel sets are bounded)
        """
        with torch.no_grad():
            for layer in self.fc_layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0)  
                    
            self.rl_layers[-1].weight.fill_(0.)
    
    def pgd(self, gamma):
        """ Scales the biases (and box constraints) by containment ratio (gamma) 
        to ensure the the feasible region of ICNN is within the true feasible set
        """
        with torch.no_grad():
            for layer in self.rl_layers:
                layer.bias.data = layer.bias.data * gamma
            self.bcl.pmax.data = self.bcl.pmax.data * gamma
            self.bcl.pmin.data = self.bcl.pmin.data * gamma

