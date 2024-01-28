# def euler(z0, t0, t1, f):
#     """
#     Simplest Euler ODE initial value solver
#     """
#     h_max = 0.05
#     n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

#     h = (t1 - t0)/n_steps
#     t = t0
#     z = z0

#     for i_step in range(n_steps):
#         z = z + h * f(z, t)
#         t = t + h
#     return z


def euler_solver(x, f, dt):
    print(len(f(x)))
    return x + dt * f(x)


