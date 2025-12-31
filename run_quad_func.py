import numpy as np
import torch
import schedulefree

from problem import get_problem
from model_set_quad_func import SetModel

# -----------------------------------------------------------------------------
ins_list = ['quadratic_func']

method_list = ['stch-set', 'tch-set', 'som', 'stch', 'tch', 'ls']

# number of independent runs
n_run = 10 
# number of optimization steps
n_steps = 10000 

# number of dimension
n_dim = 10
# number of objective
n_obj = 128

# number of solutions K
n_sol = 5 


# device
device = 'cpu'

# -----------------------------------------------------------------------------

for test_ins in ins_list:
    print(test_ins)
    
    for method in method_list:
        average_performance_list = []
        worst_performance_list = []
        
        for run_iter in range(n_run):
            print(run_iter)
            
            # get problem info
            problem = get_problem(test_ins, n_dim, n_obj, seed = run_iter)
            n_dim = problem.n_dim
            n_obj = problem.n_obj
            
        
            solution_set_model = SetModel(n_sol, n_dim)
            solution_set_model.to(device)
                
            # optimizer
            optimizer = schedulefree.SGDScheduleFree(solution_set_model.parameters(), lr=1e-2, warmup_steps = 100)
          
           
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_sol)
            pref_vec  = torch.tensor(pref).to(device).float() 
            
            
            solution_set_model.train()
            optimizer.train()
            
            for t_step in range(n_steps):

                solution_set = solution_set_model()
                value_set = problem.evaluate(solution_set)  
                
                
                # STCH-Set
                if method == 'stch-set':
                    mu =  0.1 
                    value = mu* torch.logsumexp(-mu * torch.logsumexp(-value_set / mu, dim = 0) / mu, dim = 0)  
                
              
                # TCH-Set
                if method == 'tch-set':
                    value =  torch.max(torch.min(value_set, axis = 0)[0]) 
                    
                # SOM: Sum-of-Minimization
                if method == 'som':
                    value =  torch.mean(torch.min(value_set, axis = 0)[0])
                
                    
                # STCH
                if method == 'stch':
                    mu =  0.01
                    value = mu* torch.logsumexp(pref_vec *  value_set / mu, axis = 1)  
                    value = torch.sum(value)
                    
                #TCH
                if method == 'tch':
                    value =  torch.max(pref_vec * value_set, axis = 1)[0]  
                    value = torch.sum(value)
                    
                #LS
                if method == 'ls':
                    value =  torch.sum(pref_vec * value_set , axis = 1)
                    value = torch.sum(value)
                   
                optimizer.zero_grad()
                value.backward()
                
                optimizer.step()  
                    
             
            solution_set_model.eval()
            optimizer.eval()
            
            with torch.no_grad():
                
                generated_pf = []
                generated_ps = []
                
             
                sol = solution_set_model()
                obj = problem.evaluate(sol)
        
               
                best_values = torch.min(obj, axis = 0)[0]
                average_performance = torch.mean(best_values)
                worst_performance = torch.max(best_values)
                
                average_performance_list.append(average_performance)
                worst_performance_list.append(worst_performance)
                
        mean_average_performance = np.mean(average_performance_list)
        mean_worst_performance = np.mean(worst_performance_list)
        
        print("method:", method)
        print("worse performance:", "{:.2e}".format(mean_worst_performance))
        print("average performance:", "{:.2e}".format(mean_average_performance))
    
    
    print("************************************************************")