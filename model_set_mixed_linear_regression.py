import torch
import torch.nn as nn

class SetModel(torch.nn.Module):
    def __init__(self, n_sol, n_dim):
        super(SetModel, self).__init__()
        self.n_sol = n_sol
        self.n_dim = n_dim
        
        self.SolutionSet = nn.Parameter(data=torch.randn((n_sol, n_dim), dtype=torch.float), requires_grad=True)
       
    def forward(self):
        return self.SolutionSet