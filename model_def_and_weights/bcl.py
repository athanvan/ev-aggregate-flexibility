import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class BoxConstraintLayer(nn.Module):
    """
    Implements a layer that penalizes box constraints on the inputs
    """

    def __init__(self, pmin, pmax, c=1):
        super(BoxConstraintLayer, self).__init__()
        self.pmax = nn.Parameter(pmax.detach().clone().float())
        self.pmin = nn.Parameter(pmin.detach().clone().float())
        self.c = c

    def forward(self, x):
        # calculates how far out of the box a point is & uses -inf for all points inside box
        dist = self.c*torch.sum(torch.maximum(x- self.pmax, torch.zeros_like(x)) 
                                + torch.maximum(self.pmin - x, torch.zeros_like(x)), dim=1)
        dist = torch.where(dist == 0, torch.tensor(-float('inf')), dist)
        return dist