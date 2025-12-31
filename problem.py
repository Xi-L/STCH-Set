import torch

device = 'cpu'

def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'mixed_nonlinear_regression': Mixed_Nonlinear_Regression,
        'mixed_linear_regression': Mixed_Linear_Regression,
        'quadratic_func': Quadratic_Func,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)



class Quadratic_Func():
    def __init__(self, n_dim = 10, n_obj = 128, seed = 0):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.seed = seed
        
    def evaluate(self, x):
        
        torch.manual_seed(seed=self.seed)
        
        A = 2 * torch.rand(self.n_obj, self.n_dim) - 1
        b = 4 * torch.rand(self.n_obj, 1) - 2
        
        objs = (torch.matmul(A,x.T) - b) ** 2
        
        return objs.T
    
    
class Mixed_Linear_Regression():
    def __init__(self, n_dim = 10, n_obj = 1000, n_center = 5, sigma = 0.1, seed = 0):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.seed = seed
        self.sigma = sigma
        self.n_center = n_center 
        self.x_center = torch.randn(self.n_center, self.n_dim) 
        
    def evaluate(self, x):
        
        torch.manual_seed(seed=self.seed)
        
        x_center = 4 * torch.rand(self.n_center, self.n_dim) - 2 
        self.x_center = x_center
        x_center_repeated = x_center.repeat(int(self.n_obj / self.n_center) + 1, 1)[:self.n_obj,:]
        
        A = 2 * torch.rand(self.n_obj, self.n_dim) - 1
        

        b = torch.sum(A * x_center_repeated, axis = 1) + self.sigma * torch.randn(self.n_obj)
        b = b[:,None]
        
        objs = (torch.matmul(A,x.T) - b) ** 2 
        
        return objs.T
    

class Mixed_Nonlinear_Regression():
    def __init__(self, n_dim = 10, n_obj = 1000, n_center = 5, n_hidden = 10, sigma = 0.1, seed = 0):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.seed = seed
        self.sigma = sigma
        self.n_center = n_center 
        self.n_hidden = n_hidden
        
        
    def evaluate(self, x):
        
        torch.manual_seed(seed=self.seed)
        
        W_center = torch.randn(self.n_center, self.n_hidden, self.n_dim) 
        p_center = torch.randn(self.n_center, self.n_hidden) 
        q_center = torch.randn(self.n_center, self.n_hidden) 
        o_center = torch.randn(self.n_center,1) 
        
        
        W_center_repeated = W_center.repeat(int(self.n_obj / self.n_center) + 1, 1, 1)[:self.n_obj,:,:]
        p_center_repeated = p_center.repeat(int(self.n_obj / self.n_center) + 1, 1)[:self.n_obj,:]
        q_center_repeated = q_center.repeat(int(self.n_obj / self.n_center) + 1, 1)[:self.n_obj,:]
        o_center_repeated = o_center.repeat(int(self.n_obj / self.n_center) + 1, 1)[:self.n_obj,:]
        
        A = torch.randn(self.n_obj, self.n_dim) 
        
        g = torch.sum(p_center_repeated * torch.nn.functional.relu(torch.squeeze(torch.matmul(W_center_repeated, A[:,:,None])) + q_center_repeated), axis = 1)[:,None] +  o_center_repeated
        b = g + self.sigma * torch.randn(self.n_obj)[:,None]
       
        # (n_sol, n_hidden, n_dim), (n_sol, n_hidden), (n_sol, n_hidden), (n_sol, 1)
        W, p, q, o = x[0], x[1], x[2], x[3]
        
        
        output = torch.sum(p[:,:,None] * torch.nn.functional.relu(torch.matmul(W, A.T) + q[:,:,None]), axis = 1) +  o
        
        objs = (output.T - b) ** 2 
        
        return objs.T