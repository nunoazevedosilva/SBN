# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:20:50 2022

@author: NonlinearOpticalLab
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from pylab import *
from scipy import *
from . af_loader import *
from scipy import signal
from scipy.special import jv, jn



#a simple gaussian function in 2d
def gaussian_2d_field(x_af,y_af,A,w,x0,y0,vx,vy):
    new_field = A*af.exp(-((x_af - x0)*(x_af - x0)+ (y_af - y0)*(y_af - y0)) / 
                                    (w*w))*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    return new_field


def gaussian_2d_field_int(x_af,y_af,A,w,x0,y0,vx,vy):
    new_field = A*af.exp(-2*((x_af - x0)*(x_af - x0)+ (y_af - y0)*(y_af - y0)) / 
                                    (w*w))*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    return new_field


#a simple gaussian function in 2d
def bessel_2d_field_old(x_af,y_af,A,w,x0,y0):
    r=af.sqrt(((x_af - x0)*(x_af - x0)+ (y_af - y0)*(y_af - y0))/ (w*w))
    new_field = special.jv(0,r)
    return new_field

def snake(A,x_af,y_af,x0,delta):
    new_field = A
    #new_field+= -A*(x_af<(x0+delta))*(x_af>(x0-delta))
    new_field+= -2*A*(x_af>(x0+delta))
    return new_field

def planewave_2d_field(x_af,y_af,A,vx):
    new_field = A*af.exp(1.0j*(vx*(x_af)))
                         
    return new_field

def whitenoise_2d_field(x_af,A):
    new_field = A*af.randu(x_af.dims()[0], x_af.dims()[1])*af.exp(1.0j*2*pi*af.randu(x_af.dims()[0],x_af.dims()[1]))
                         
    return new_field

def bessel_2d_field_old(x_af,y_af,A,x0,y0,d):
    v = linspace(0, 2*pi, 1000)
    dv = v[1] - v[0]
       
    fz = 2.4048
    fs = fz / (d/2)
    
    r = af.sqrt((x_af-x0)*(x_af-x0) + (y_af-y0)*(y_af-y0))
    resultado = fs*r
    
    my_field = resultado*0.
    for k in v:
        my_field += af.exp(1.0j * resultado * np.cos(k))*dv
       
    return (A*(af.pow(af.abs(my_field)/(2*pi),2))) # * (r <= 2*d)


def bessel_2d(x_af, y_af, A, x0, y0, d):
   
    np_array =  A*(jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d ))**1.0 
    return af.to_array(np_array)

def add_field_from_array_with_velocity(array_to_add, x_af, y_af, A, x0, y0, vx, vy):
    new_field = af.interop.from_ndarray(A*array_to_add)*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    return new_field