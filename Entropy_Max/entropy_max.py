import sys
sys.path.append('/work/mccleary_group/habjan.e/TNG/TNG_cluster_dynamics')

import TNG_DA
import numpy as np 
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import torch

### Import data for a cluster
pos, vel, group, sub_masses_TNG, h, halo_mass_TNG = TNG_DA.get_cluster_props(3)

### Project Data to observation-like data
pos_2d, vel_los = TNG_DA.project_3d_to_2d(pos, vel)

### Input parameters
N = len(pos_2d)
masses = np.array([10**10 for i in range(len(pos_2d))]) 

T = 10

x_pos = pos[:, 0]
y_pos = pos[:, 1]
z_pos = np.random.uniform(np.min(pos_2d), np.max(pos_2d), size=(N))

x_vel = np.random.uniform(np.min(vel_los), np.max(vel_los), size=(N))
y_vel = np.random.uniform(np.min(vel_los), np.max(vel_los), size=(N))
z_vel = vel[:, 2] / (3.086 * 10**16)

p0 = np.hstack([masses, z_pos, x_vel, y_vel, T])

### PyTorch version of probability functions
def zeroth_order_Q_i(m, q_ix, q_iy, q_iz, beta, h):

    prefactor = (2 * torch.pi * m / (beta * h**2))**(1.5)
    Q_i_raw = prefactor * q_ix * q_iy * q_iz
    Q_i = torch.log(torch.clamp(Q_i_raw, min=10**-3))

    return Q_i

def zeroth_order_Q_total(masses, x_pos, y_pos, z_pos, beta, h):
    N = len(masses)
    Qi_arr = torch.stack([
        zeroth_order_Q_i(masses[i], x_pos[i], y_pos[i], z_pos[i], beta, h)
        for i in range(N)
    ])
    # Filter out inf/nan/0
    mask = (~Qi_arr.isnan()) & (~Qi_arr.isinf()) & (Qi_arr != 0)
    Q = torch.abs(torch.sum(Qi_arr[mask]))
    return Q

def probability_func(init_guess, x_pos, y_pos, z_vel):
    N = len(x_pos)

    gal_m = torch.clamp(init_guess[0:N], min=10**-3)
    z_pos = init_guess[N:2*N]
    x_vel = init_guess[2*N:3*N] / (3.086 * 10**16)
    y_vel = init_guess[3*N:4*N] / (3.086 * 10**16)
    temp = torch.clamp(init_guess[-1], min=10**-3)

    G_si = 6.67e-11
    h_si = 6.626e-34
    k_B_si = 1.38e-23

    k_B = k_B_si * (1 / (1.989e30)) * (1 / (3.086e19)**2)
    h = h_si * (1 / (1.989e30)) * (1 / (3.086e19)**2)
    G = G_si * (1.989e30) * (1 / (3.086e19)**3)

    beta = 1 / (k_B * temp)

    Q = zeroth_order_Q_total(gal_m, x_pos, y_pos, z_pos, beta, h)
    probabilities = []

    for i in range(N):
        m_r_list = []
        for j in range(i+1, N):
            dx = x_pos[i] - x_pos[j]
            dy = y_pos[i] - y_pos[j]
            dz = z_pos[i] - z_pos[j]
            r = torch.sqrt(dx**2 + dy**2 + dz**2)
            r_clamp = torch.clamp(r, min = 10**-3)
            m_r_list.append(gal_m[j] / r)
        m_r_sum = torch.nansum(torch.stack(m_r_list), dtype = torch.float64) if m_r_list else torch.tensor(0.0, dtype = torch.float64, device=init_guess.device)

        ke_i = gal_m[i] * (x_vel[i]**2 + y_vel[i]**2 + z_vel[i]**2) / 2
        pe_i = - ((G * gal_m[i]) / 2) * m_r_sum

        log_state = -beta * (ke_i + pe_i)
        probabilities.append(-Q + log_state)

    prob_tensor = torch.stack(probabilities)

    likelihood = torch.logsumexp(prob_tensor, dim=0)

    neg_likelihood = -likelihood
    
    return neg_likelihood 

### Pytorch optimization
dtype = torch.float64

x_pos_ten = torch.tensor(x_pos, dtype = dtype)
y_pos_ten = torch.tensor(y_pos, dtype = dtype)
z_vel_ten = torch.tensor(z_vel, dtype = dtype)
p0_ten = torch.tensor(p0, dtype = dtype, requires_grad=True)

optimizer = torch.optim.Adam([p0_ten], lr=10**-1)

iters = 10**3

for step in range(iters):
    optimizer.zero_grad()
    loss = probability_func(p0_ten, x_pos_ten, y_pos_ten, z_vel_ten)
    loss.backward()
    optimizer.step()
    #if step % 10 == 0:
    print(f"Step {step} | Loss: {loss.item()}")

data_path = '/home/habjan.e/TNG/Sandbox_notebooks/entropy_max_results/'

gal_m_result = p0_ten[0:len(x_pos)].detach().numpy()
z_pos_result = p0_ten[len(x_pos):2*len(x_pos)].detach().numpy()
x_vel_result = p0_ten[2*len(x_pos):3*len(x_pos)].detach().numpy() #* (3.086 * 10**16)
y_vel_result = p0_ten[3*len(x_pos):4*len(x_pos)].detach().numpy() #* (3.086 * 10**16)
T_result = p0_ten[-1].detach().numpy()

np.save(data_path + 'gal_m_result.npy', gal_m_result)
np.save(data_path + 'z_pos_result.npy', z_pos_result)
np.save(data_path + 'x_vel_result.npy', x_vel_result)
np.save(data_path + 'y_vel_result.npy', y_vel_result)
np.save(data_path + 'T_result.npy', T_result)

np.save(data_path + 'z_pos_guess.npy', z_pos)
np.save(data_path + 'x_vel_guess.npy', x_vel)
np.save(data_path + 'y_vel_guess.npy', y_vel)