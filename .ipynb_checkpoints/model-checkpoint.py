import torch
import numpy as np
import torch.nn as nn
from mass_spring import get_dataset
from tqdm import tqdm
from torch.autograd import grad
torch.manual_seed(42)
np.random.seed(42)

from torch.optim.lr_scheduler import ReduceLROnPlateau

class MLP1H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP1H_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)

    def kinetic_energy(self, p):
        h_pre = self.linear_K1(p)
        h = h_pre.tanh_()
        return self.linear_K2(h)

    def potential_energy(self, q):
        h_pre = self.linear_P1(q)
        h = h_pre.tanh_()
        return self.linear_P2(h)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)
    
class MLP2H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP2H_Separable_Hamilt, self).__init__()
        self.linear_K1 = nn.Linear(input_size, n_hidden)
        self.linear_K1B = nn.Linear(n_hidden, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, 1)
        self.linear_P1 = nn.Linear(input_size, n_hidden)
        self.linear_P1B = nn.Linear(n_hidden, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, 1)

    def kinetic_energy(self, p):
        h_pre = self.linear_K1(p)
        h = h_pre.tanh_()
        h_pre_B = self.linear_K1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear_K2(h_B)

    def potential_energy(self, q):
        h_pre = self.linear_P1(q)
        h = h_pre.tanh_()
        h_pre_B = self.linear_P1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear_P2(h_B)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)


class MLP3H_Separable_Hamilt(nn.Module):
    def __init__(self, n_hidden, input_size):
        super(MLP3H_Separable_Hamilt, self).__init__()
        self.linear_K0 = nn.Linear(input_size, n_hidden)
        self.linear_K1 = nn.Linear(n_hidden, n_hidden)
        self.linear_K2 = nn.Linear(n_hidden, n_hidden)
        self.linear_K3 = nn.Linear(n_hidden, 1)
        self.linear_P0 = nn.Linear(input_size, n_hidden)
        self.linear_P1 = nn.Linear(n_hidden, n_hidden)
        self.linear_P2 = nn.Linear(n_hidden, n_hidden)
        self.linear_P3 = nn.Linear(n_hidden, 1)

    def kinetic_energy(self, p):
        h = self.linear_K0(p).tanh_()
        h = self.linear_K1(h).tanh_()
        h = self.linear_K2(h).tanh_()
        return self.linear_K3(h)

    def potential_energy(self, q):
        h = self.linear_P0(q).tanh_()
        h = self.linear_P1(h).tanh_()
        h = self.linear_P2(h).tanh_()
        return self.linear_P3(h)

    def forward(self, p, q):
        return self.kinetic_energy(p) + self.potential_energy(q)
        
class MLP_General_Hamilt(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(MLP_General_Hamilt, self).__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear1B = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, p, q):
        pq = torch.cat((p, q), 1)
        h_pre = self.linear1(pq)
        h = h_pre.tanh_()
        h_pre_B = self.linear1B(h)
        h_B = h_pre_B.tanh_()
        return self.linear2(h_B)

def leapfrog(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    if is_Hamilt:
        hamilt = Func(p, q)
        dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p_half, q)
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            q_next = q + dqdt * dt

            hamilt = Func(p_half, q_next)
            dpdt = -grad(hamilt.sum(), q_next, create_graph=not volatile)[0]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]
        time_drvt = Func(torch.cat((p, q), 1))
        dpdt = time_drvt[:, :dim]

        for i in range_of_for_loop:
            p_half = p + dpdt * (dt / 2)

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(torch.cat((p_half, q), 1))
            dqdt = time_drvt[:, dim:]

            q_next = q + dqdt * dt

            time_drvt = Func(torch.cat((p_half, q_next), 1))
            dpdt = time_drvt[:, :dim]

            p_next = p_half + dpdt * (dt / 2)

            p = p_next
            q = q_next

    return trajectories


def euler(p_0, q_0, Func, T, dt, volatile=True, is_Hamilt=True, device='cpu', use_tqdm=False):
    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)
    # with torch.enable_grad():

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    if is_Hamilt:

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q

            hamilt = Func(p, q)
            dpdt = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
            dqdt = grad(hamilt.sum(), p, create_graph=not volatile)[0]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p

            time_drvt = Func(torch.cat((p, q), 1))
            dpdt = time_drvt[:, :dim]
            dqdt = time_drvt[:, dim:]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    return trajectories

def predictor(p, q, Func, dt, volatile):
    hamilt = Func(p, q)
    k1 = grad(hamilt.sum(), p, create_graph=not volatile)[0]
    l1 = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
    q_next = q + dt*q
    p_next = p + dt*p
    hamilt = Func(p, q_next)
    k2 = grad(hamilt.sum(), p, create_graph=not volatile)[0]
    hamilt = Func(p_next, q)
    l2 = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
    q_next = q + dt * 0.5 * (k1 + k2)
    p_next = p + dt * 0.5 * (l1 + l2)
    return p_next, q_next
    
    # hamilt = Func(p, q)
    # k1 = -grad(hamilt.sum(), q, create_graph=not volatile)[0]
    # p_next = p + dt*p
    # hamilt = Func(p_next, q)
    # k2 = grad(hamilt.sum(), q, create_graph=not volatile)[0]
    # p_next = p + dt * 0.5 * (k1 + k2)
    return p_next, q_next

def corrector(p_n, q_n, p_n_1, q_n_1, Func, dt, iterations, volatile):
    for _ in range(iterations):
        mid_point = (p_n + p_n_1)/2
        hamilt = Func(mid_point, q_n)
        k1 = grad(hamilt.sum(), mid_point, create_graph=not volatile)[0]
        l1 = -grad(hamilt.sum(), q_n, create_graph=not volatile)[0]
        hamilt = Func(mid_point, q_n_1)
        k2 = grad(hamilt.sum(), mid_point, create_graph=not volatile)[0]
        l2 = -grad(hamilt.sum(), q_n_1, create_graph=not volatile)[0]
        q_next = q_n + dt * 0.5 * (k1 + k2)
        p_next = p_n + dt * 0.5 * (k1 + k2)
        p_n_1 = p_next
        q_n_1 = q_next
        
    return p_n_1, q_n_1
    

def predictor_corrector(p_0, q_0, Func, T, dt, volatile = True, is_Hamilt = True, device = 'cpu', use_tqdm = False, corrector_iterations = 1):

    trajectories = torch.empty((T, p_0.shape[0], 2 * p_0.shape[1]), requires_grad=False).to(device)

    p = p_0
    q = q_0
    p.requires_grad_()
    q.requires_grad_()

    if use_tqdm:
        range_of_for_loop = tqdm(range(T))
    else:
        range_of_for_loop = range(T)

    if is_Hamilt:

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :p_0.shape[1]] = p.detach()
                trajectories[i, :, p_0.shape[1]:] = q.detach()
            else:
                trajectories[i, :, :p_0.shape[1]] = p
                trajectories[i, :, p_0.shape[1]:] = q
            p_next, q_next = predictor(p, q, Func, dt)
            p_next, q_next = corrector(p, q, p_next, q_next, Func, dt, corrector_iterations)
            p = p_next
            q = q_next

    else:
        dim = p_0.shape[1]

        for i in range_of_for_loop:

            if volatile:
                trajectories[i, :, :dim] = p.detach()
                trajectories[i, :, dim:] = q.detach()
            else:
                trajectories[i, :, :dim] = p
                trajectories[i, :, dim:] = q

            time_drvt = Func(torch.cat((p, q), 1))
            dpdt = time_drvt[:, :dim]
            dqdt = time_drvt[:, dim:]

            p_next = p + dpdt * dt
            q_next = q + dqdt * dt

            p = p_next
            q = q_next

    return trajectories


def numerically_integrate(integrator, p_0, q_0, model, method, T, dt, volatile, device, coarsening_factor=1):
    print("inside numerically_integrate")
    
    if (coarsening_factor > 1):
        fine_trajectory = numerically_integrate(integrator, p_0, q_0, model, method, T * coarsening_factor, dt / coarsening_factor, volatile, device)
        print("fine_trajectory", fine_trajectory.shape)
        trajectory_simulated = fine_trajectory[np.arange(T) * coarsening_factor, :, :]
        print("trajectory_simulated", trajectory_simulated.shape)
        return trajectory_simulated
    if (method == 5):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, device=device)
        elif (integrator == 'predictor_corrector'):
            trajectory_simulated = predictor_corrector(p_0, q_0, model, T, dt, volatile=volatile, device=device)
    elif (method == 1):
        if (integrator == 'leapfrog'):
            trajectory_simulated = leapfrog(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
        elif (integrator == 'euler'):
            trajectory_simulated = euler(p_0, q_0, model, T, dt, volatile=volatile, is_Hamilt=False, device=device)
    else:
        trajectory_simulated = model(torch.cat([p_0, q_0], dim=1), T)
    return trajectory_simulated

def predict(test_data, model, method, integrator, device, T_test, dt):
    print(integrator)
    dim = 20
    model = model.to(device).to(dtype=torch.float64)
    model.eval()

    z_0_npy = test_data[0, :, :]

    z_0 = torch.from_numpy(z_0_npy).to(device).to(dtype=torch.float64)

    trajectory_predicted = numerically_integrate(integrator=integrator, p_0=z_0[:, :dim], q_0=z_0[:, dim:], model=model, \
        method=method, T=T_test, dt=dt, volatile=True, device=device, coarsening_factor=1).to(torch.float64)

    return trajectory_predicted