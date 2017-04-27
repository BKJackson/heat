# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:35:16 2017

@author: Ondrej Lexa

Modul pro reseni 1D termalni rovnice
s Dirichletovskou podminkou nahore
a Neumannovou podminkou dole

Priklad:
from heatlib_var import *
m = dict(n=100, k=2.25*np.ones(99), H=1e-6*np.ones(99), tc=35000, T0=0, q=-0.02)
init(m)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

ysec = 365.25*24*3600
ma = 1e6*ysec

def tshow(m):
    plt.plot(m['t'], m['x'], 'b')
    plt.ylim(m['tc'], 0)
    plt.xlabel('Teplota')
    plt.ylabel('Hloubka')
    plt.title('Čas: {:.2f} Ma  Tepelný tok na povrchu: {:.2f} mW/m2'.format(m['time']/ma, 1000 * q_surface(m)))

def kappa(m):
    return m['k'] / (m['rho'] * m['c'])

def q_surface(m):
    return m['k'][0] * (m['t'][1] - m['t'][0]) / m['dx']

def max_dt(m):
    return m['dx']**2 / (2 * kappa(m))

def init(m):
    """ Stacionarni reseni a inicializace"""
    m['x'] = np.linspace(0, m['tc'], m['n'])
    m['dx'] = m['x'][1] - m['x'][0]
    m['xm'] = np.linspace(m['dx']/2, m['tc'] - m['dx']/2, m['n']-1)
    # Sestaveni diagonal matice soustavy
    kd = np.hstack((m['k'][:-1], 2 * m['k'][-1], 0))
    ku = np.hstack((0, 0, m['k'][1:]))
    km = np.hstack((1, -(m['k'][:-1] + m['k'][1:]),  -2 * m['k'][-1]))
    # matice soustavy
    A = spdiags([kd, km, ku], [-1, 0, 1], m['n'], m['n'], 'csr')
    # vektor pravy strany
    b = np.hstack((0, -(m['H'][:-1] + m['H'][1:])*m['dx']**2/2, m['q']*2*m['dx'] - m['H'][-1]*m['dx']**2))
    # reseni
    m['time'] = 0
    m['t'] = spsolve(A, b)

