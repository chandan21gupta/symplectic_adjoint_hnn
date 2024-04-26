import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from tqdm import tqdm
torch.manual_seed(42)
import random
from abc import ABC, abstractmethod
random.seed(0)
np.random.seed(0)

class Solver(ABC, nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class EulerSolver(Solver):
    def __init__(self):
        super(EulerSolver, self).__init__()
    
    def forward(self, dynamics_fn, p0, q0, dt):
        dp_dt, dq_dt = dynamics_fn(p0, q0)
        p = p0 + dt * dp_dt
        q = q0 + dt * dq_dt
        return p, q
    
class LeapFrogSolver(Solver):
    def __init__(self):
        super(LeapFrogSolver, self).__init__()
        
    def forward(self, dynamics_fn, p0, q0, dt):
        p_half = p0 + 0.5 * dt * dynamics_fn(p0, q0)[0]
        q = q0 + dt * dynamics_fn(p_half, q0)[1]
        p = p_half + 0.5 * dt * dynamics_fn(p_half, q)[0]
        return p, q
    
    
class RK2Solver(Solver):
    def __init__(self):
        super(RK2Solver, self).__init__()
        
    def forward(self, dynamics_fn, p0, q0, dt):
        p1, q1 = dynamics_fn(p0, q0)
        p2, q2 = dynamics_fn(p0 + 0.5 * dt * p1, q0 + 0.5 * dt * q1)
        p = p0 + dt * (p1 + p2) / 2 
        q = q0 + dt * (q1 + q2) / 2
        return p, q
    
class SVSolver(Solver):
    
    def __init__(self):
        super(SVSolver, self).__init__()
        
    def forward(self, dynamics_fn, p0, q0, dt, iterations = 10, p_init = None, q_init = None):
        if p_init:
            p_half = (p_init + p0)/2
        else:
            p_half = p0
        for _ in range(iterations):
            p_half = p0 + 0.5 * dt * dynamics_fn(p_half, q0)[0]
        q1 = q0 + 0.5 * dt * dynamics_fn(p_half, q0)[1]
        if q_init:
            q2 = (q_init + q1)/2
        else:
            q2 = q1
        for _ in range(iterations):
            q2 = q1 + 0.5 * dt * dynamics_fn(p_half, q2)[1]
        p1 = p_half + 0.5 * dt * dynamics_fn(p_half, q2)[0]
        return p1, q2
    
class PCSolver(Solver):
    
    def __init__(self):
        super(PCSolver, self).__init__()
        
    def forward(self, dynamics_fn, p0, q0, dt, implicit, explicit, iterations):
        p, q = explicit.step(dynamics_fn, p0, q0, dt)
        p, q = implicit.step(dynamics_fn, p0 = p0, q0 = q0, dt = dt, iterations = iterations, p_init = p, q_init = q)
        return p, q