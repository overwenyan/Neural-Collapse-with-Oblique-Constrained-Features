
import autograd.numpy as np
import matplotlib.pyplot as plt

from autograd import grad
from autograd import value_and_grad 
from pymanopt.manifolds import Oblique
from pymanopt.solvers import SteepestDescent
# from mpl_toolkits.mplot3d import Axes3D

import re
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from helpers.oblique import *
from helpers.simulations import *

if __name__ == '__main__':
    # (1) Instantiate a manifold
    # manifold = Stiefel(5, 2)
    # (2) Define the cost function (here using autograd.numpy)
    # def cost(X): return np.sum(X)
    # problem = Problem(manifold=manifold, cost=cost)
    # (3) Instantiate a Pymanopt solver
    # solver = SteepestDescent()
    # let Pymanopt do the rest
    # Xopt = solver.solve(problem)
    # print(Xopt)

    num_cls = 10    # class count
    sample_per_cls = 5 
    N = sample_per_cls * num_cls
    M = 2           # dim of feature??
    solver = TrustRegions()
    manifold = Product((Euclidean(M, num_cls), Euclidean(M, N), Euclidean(num_cls)))
    Xopt, data = solve_prob(make_lr_weight_decay(0.1, 0, 0.01), solver, manifold)

    W, H, b = Xopt
    # print W(dim=2)
    fig, axis = plt.subplots()
    axis.scatter(W[0, :], W[1, :])
    axis.grid(True)
    axis.set_aspect("equal")
    plt.show()
    # plt.savefig(f"images/{num_cls}_cls_2d_WH_ob_weights.png")
    
    check_etf(Xopt, verbose=True)


    