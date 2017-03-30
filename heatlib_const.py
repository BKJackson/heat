# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:08:54 2012

@author: Ondrej Lexa

Modul pro reseni 1D termalni rovnice
s Dirichletovskou podminkou nahore
a Neumannovou podminkou dole

Priklad:
m = dict(n=100, k=2.25, H=1e-6, rho=2700,
         c=800, tc=35000, T0=0, q=-0.02)
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
    return m['k'] * (m['t'][1] - m['t'][0]) / m['dx']

def max_dt(m):
    return m['dx']**2 / (2 * kappa(m))

def init(m):
    """ Stacionarni reseni a inicializace"""
    m['x'] = np.linspace(0, m['tc'], m['n'])
    m['dx'] = m['x'][1] - m['x'][0]
    d = np.ones_like(m['x'])
    # matice soustavy
    A = spdiags([d, -2*d, d], [-1, 0, 1], m['n'], m['n'], 'csr')
    # vektor pravy strany
    b = -d * m['H'] * m['dx']**2 / m['k']
    # Okrajove podminky
    A[0, :2] = [1, 0]
    b[0] = m['T0']
    A[-1, -2:] = [2, -2]
    b[-1] += 2 * m['q'] * m['dx'] / m['k']
    # reseni
    m['time'] = 0
    m['t'] = spsolve(A, b)

def ftcs(m, dt):
    """ Evolucni reseni schema FTCS """
    d = np.ones_like(m['x'])
    u = kappa(m) * dt / m['dx']**2
    # matice soustavy
    A = spdiags([d*u, d*(1 - 2*u), d*u], [-1, 0, 1], m['n'], m['n'], 'csr')
    #vektor b
    b = d * m['H'] * dt / (m['rho'] * m['c'])
    
    #Okrajove podminky
    A[0, :2] = [1, 0]
    b[0] = m['T0']
    A[-1, -2] = 2*u
    b[-1] = dt*(m['H']*m['dx'] - 2*m['q']) / (m['rho'] * m['c']*m['dx'])
    
    # reseni
    m['time'] += dt
    m['t'] = A.dot(m['t']) + b
