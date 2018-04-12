# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:23:34 2018

@author: Ondrej Lexa

1D heat equation solver with
variable thermal properties on
regular grid

Example:
from heatlib_var import *
bc_surf = dict(kind='dirichlet', val=0)
bc_moho = dict(kind='neumann', val=-0.02)

m = dict(n=100, k=2.25*np.ones(99), H=1e-6*np.ones(99),
         rho=2700*np.ones(99), c=800*np.ones(99), tc=35000,
         bc0=bc_surf, bc1=bc_moho)
init(m)
m['t'][(m['x']>10000) & (m['x']<15000)] = 700
tshow(m)

for i in range(100):
    btcs(m, 10000*ysec)

tshow(m)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

hour = 3600
day = 24*hour
year = 365.25*day
ma = 1e6*year

def tshow(m):
    plt.plot(m['t'], m['x'], 'b')
    plt.ylim(m['tc'], 0)
    plt.xlabel('Temperature')
    plt.ylabel('Depth')
    plt.title('Time: {:.2f} Ma  HF0: {:.2f} mW/m2  HF1: {:.2f} mW/m2'.format(m['time']/ma, -1000 * q_0(m), -1000 * q_1(m)))
    plt.axis('tight')
    plt.show()

def q_0(m):
    return -m['k'][0] * (m['t'][1] - m['t'][0]) / m['dx']

def q_1(m):
    return -m['k'][-1] * (m['t'][-1] - m['t'][-2]) / m['dx']

def get_q(m, x):
    l = np.hstack((m['k'][0], (m['k'][:-1] + m['k'][1:])/2, m['k'][-1]))
    return np.interp(x, m['x'], -l*np.gradient(m['t'], m['dx']))

def get_t(m, x):
    return np.interp(x, m['x'], m['t'])

def get_x(m, t):
    return np.interp(t, m['t'], m['x'])

def init(m):
    """ Initialization and steady-state solution"""
    m['x'] = np.linspace(0, m['tc'], m['n'])
    m['dx'] = m['x'][1] - m['x'][0]
    m['xm'] = np.linspace(m['dx']/2, m['tc'] - m['dx']/2, m['n']-1)
    # Boundary conditions
    if m['bc0']['kind'] == 'dirichlet':
        u2 = [1, 0]
        bu = m['bc0']['val']
    elif m['bc0']['kind'] == 'neumann':
        u2 = [-2 * m['k'][0], 2 * m['k'][0]]
        bu = m['H'][0]*m['dx']**2 - m['bc0']['val']*2*m['dx']
    else:
        raise(Exception('BC0 has unsupported kind.'))
    if m['bc1']['kind'] == 'dirichlet':
        l2 = [1, 0]
        bl = m['bc1']['val']
    elif m['bc1']['kind'] == 'neumann':
        l2 = [-2 * m['k'][-1], 2 * m['k'][-1]]
        bl = m['bc1']['val']*2*m['dx'] - m['H'][-1]*m['dx']**2
    else:
        raise(Exception('BC1 has unsupported kind.'))
    # coefficient matrix
    du = np.hstack((0, u2[1], m['k'][1:]))
    dm = np.hstack((u2[0], -(m['k'][:-1] + m['k'][1:]),  l2[0]))
    dl = np.hstack((m['k'][:-1], l2[1], 0))
    A = spdiags([dl, dm, du], [-1, 0, 1], m['n'], m['n'], 'csr')
    # load vector
    b = np.hstack((bu, -(m['H'][:-1] + m['H'][1:])*m['dx']**2/2, bl))
    # solve
    m['time'] = 0
    m['t'] = spsolve(A, b)

def btcs(m, dt):
    """ Evolucni reseni schema BTCS """
    # helpers
    li = m['k'][:-1]/m['dx']**2
    ri = m['k'][1:]/m['dx']**2
    mi = (m['rho'][:-1] + m['rho'][1:]) * (m['c'][:-1] + m['c'][1:]) / (4*dt)
    # Sestaveni diagonal matice soustavy
    # matice soustavy
        # Boundary conditions
    if m['bc0']['kind'] == 'dirichlet':
        u2 = [1, 0]
        bu = m['bc0']['val']
    elif m['bc0']['kind'] == 'neumann':
        u2 = [m['rho'][0]*m['c'][0]/dt + 2*li[0], -2*li[0]]
        bu = 2*m['bc0']['val']/m['dx'] - m['t'][0]*m['rho'][0]*m['c'][0]/dt - m['H'][0]
    else:
        raise(Exception('BC0 has unsupported kind.'))
    if m['bc1']['kind'] == 'dirichlet':
        l2 = [1, 0]
        bl = m['bc1']['val']
    elif m['bc1']['kind'] == 'neumann':
        l2 = [m['rho'][-1]*m['c'][-1]/dt + 2*ri[-1], -2*ri[-1]]
        bl = m['H'][-1]-2*m['bc1']['val']/m['dx'] + m['t'][-1]*m['rho'][-1]*m['c'][-1]/dt
    else:
        raise(Exception('BC1 has unsupported kind.'))
    # coefficient matrix
    du = np.hstack((0, u2[1], -ri))
    dm = np.hstack((u2[0], mi + ri + li, l2[0]))
    dl = np.hstack((-li,  l2[1], 0))
    A = spdiags([dl, dm, du], [-1, 0, 1], m['n'], m['n'], 'csr')
    # vektor pravy strany
    Hm = (m['H'][:-1] + m['H'][1:])/2
    b = np.hstack((bu, Hm + mi*m['t'][1:-1], bl))
    # reseni
    m['time'] += dt
    m['t'] = spsolve(A, b)

