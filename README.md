# STCH-Set

Code for ICLR2025 Paper: Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization

The code is mainly designed to be simple and readable, it contains:
- <code>run_[quad_func/model_set_mixed_linear_regression/model_set_mixed_nonlinear_regression].py</code> is a ~150-line main file to run the STCH-Set method for the problem of [convex optimization/noisy linear regression/noisy nonlinear regression];
- <code>model_set_[quad_func/model_set_mixed_linear_regression/model_set_mixed_nonlinear_regression].py</code> is a simple torch.nn.Module that stores the set solutions for the problem of [convex optimization/noisy linear regression/noisy nonlinear regression];
- <code>problem.py</code> contains all test problems used in this paper.



**Reference**

If you find our work helpful for your research, please cite our paper:
```
@inproceedings{lin2025few,
  title={Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization},
  author={Lin, Xi and Liu, Yilu and Zhang, Xiaoyuan and Liu, Fei and Wang, Zhenkun and Zhang, Qingfu},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
