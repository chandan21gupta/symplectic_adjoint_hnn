import autograd
import autograd.numpy as np
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp
    

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = p**2 + q**2 # spring hamiltonian (linear oscillator)
    return H

def leapfrog(coord, dt, dynamics_fn):
    S = dynamics_fn(1, coord)
    dpdt = S[0]
    dqdt = S[1]
    p_half = coord[1] + dqdt * (dt/2)
    q_next = coord[0] + dt * dynamics_fn(1, np.array((coord[0], p_half)))[0]
    arr = np.array([q_next, p_half])
    p_next = p_half + (dt/2) * dynamics_fn(1, arr)[1]
    return q_next, p_next

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(noise = True, t_span=[0,3], timescale=10, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if y0 is None:
        y0 = np.random.rand(2)*2-1
    if radius is None:
        radius = np.random.rand()*0.9 + 0.1 # sample a range of radii
    y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    # spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    # q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    q = [y0[0]]
    p = [y0[1]]
    # print("generating dataset from leapfrog")
    for i in range(1, timescale*(t_span[1] - t_span[0])):
        q_, p_ = leapfrog(y0, 0.01, dynamics_fn)
        q.append(q_)
        p.append(p_)
        y0 = np.array([q_, p_])
    q = np.array(q)
    p = np.array(p)
    # dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    # dydt = np.stack(dydt).T
    # dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    if noise:
        q += np.random.randn(*q.shape)*noise_std
        p += np.random.randn(*p.shape)*noise_std
    # return q, p, dqdt, dpdt, t_eval
    return q, p

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        # x, y, dx, dy, t = get_trajectory(**kwargs)
        x, y, = get_trajectory(**kwargs)
        xs.append( np.stack( [x, y]).T )
        # dxs.append( np.stack( [dx, dy]).T )
    print((xs[0].shape))
    data['x'] = np.stack(xs, axis = 0)
    # data['dx'] = np.concatenate(dxs).squeeze()

    # make a train/test split
    split_ix = int(len(data['x']) * test_split)
    split_data = {}
    for k in ['x']:
        split_data[k], split_data['test_' + k] = data[k][:split_ix], data[k][split_ix:]
    data = split_data
    return data

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field