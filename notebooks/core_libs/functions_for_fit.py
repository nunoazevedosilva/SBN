# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:33:23 2022

@author: NonlinearOpticalLab
"""

import numpy as np
from scipy.special import jv, jn

def linear_fit(x, a, b):
    return a*x+ b

def delta_n_f(I, n_max, I_sat):
    return n_max*I/(I + I_sat)

def bessel_function(x, x0, w_b):
    return (jn(0, 2.4048*(x-x0)/w_b ))**2.0

def gaus(x, I0, x0, w_z0):
    return I0*np.exp(-2.0*(x-x0)**2/(w_z0**2))

def gaus_force(x, I0, x0, w_z0):
    return I0*np.exp(-(x-x0)**2/(2.0*w_z0**2))