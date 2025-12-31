import torch
import torch.nn as nn

class SetModel(torch.nn.Module):
    def __init__(self, n_sol, n_dim, n_hidden):
        super(SetModel, self).__init__()
        self.n_sol = n_sol
        self.n_dim = n_dim
        self.n_dim = n_hidden
        
        # [W, p, q, o]
        self.SolutionSet = nn.ParameterList([nn.Parameter(torch.randn(n_sol, n_hidden, n_dim)), nn.Parameter(torch.randn(n_sol, n_hidden)), 
                                             nn.Parameter(torch.randn(n_sol, n_hidden)), nn.Parameter(torch.randn(n_sol, 1))])
       
    def forward(self):
        return self.SolutionSet