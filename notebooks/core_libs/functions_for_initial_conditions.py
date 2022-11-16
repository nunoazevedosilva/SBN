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
def gaussian_2d_field(x_af , y_af, A, w, x0, y0, vx, vy):
    new_field = A*af.exp(-((x_af - x0)*(x_af - x0) + (y_af - y0)*(y_af - y0)) / (w*w))
    new_field = new_field*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    return new_field

#a simple gaussian function in 2d
def gaussian_2d_field_power_normalized(x_af , y_af, w, x0, y0, vx, vy):
    new_field = af.exp(-((x_af - x0)*(x_af - x0) + (y_af - y0)*(y_af - y0)) / (w*w))
    new_field = new_field*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    
    norm = np.sum(np.abs(new_field)**2.0)*(x_af[1,0]-x_af[0,0])*(y_af[0,1]-  y_af[0,0])
    
    return af.to_array(new_field/np.sqrt(norm))
    


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

def whitenoise_2d_field_v2(x_af, A, B):
    noise_1 = 0.5*(2.0*af.randu(x_af.dims()[0], x_af.dims()[1]) - 1.0)
    noise_2 = 0.5*(2.0*af.randu(x_af.dims()[0], x_af.dims()[1]) - 1.0)
    
    new_field = A*noise_1*af.exp(1.0j*B*pi*noise_2)
                         
    return new_field

def bessel_2d_field_old(x_af,y_af,defect_power, defect_intensity_normalization, x0, y0, d, factor_t):
    v = linspace(0, 2*pi, 1000)
    dv = v[1] - v[0]
       
    fz = 2.4048
    fs = fz / (d/2)
    
    r = af.sqrt((x_af-x0)*(x_af-x0) + (y_af-y0)*(y_af-y0))
    resultado = fs*r
    
    my_field = resultado*0.
    for k in v:
        my_field += af.exp(1.0j * resultado * np.cos(k))*dv
       
    
    np_array = af.sqrt(((af.pow(af.abs(my_field)/(2*pi),2))))
    
    d_x = np.array(x_af[1,0]-x_af[0,0])
    d_y = np.array(y_af[0,1]-y_af[0,0])
    

    np_array1 = np.array(np_array, order="F")
    norm = np.sum(np.abs(np_array1)**2.0)*d_x*d_y/factor_t/factor_t*1e4
    
    I_defect = defect_power/norm
    
    

    I_defect_normalized = I_defect/defect_intensity_normalization

    print(type(np_array1))
    print(I_defect_normalized)
    
    return af.to_array(np.sqrt(I_defect_normalized[0])*np.array(np_array, order="F"))
    



def bessel_2d(x_af, y_af, defect_power, defect_intensity_normalization, x0, y0, d, factor_t):
   
    np_array =  (jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d ))**1.0 
    
    x_max = np.max(x_af)/factor_t*1e3
    y_max = np.max(y_af)/factor_t*1e3
    
    d_x = (x_af[1,0]-x_af[0,0])
    d_y = (y_af[0,1]-y_af[0,0])
    
    d_x *= 5.0/x_max
    d_y *= 5.0/y_max
    
    norm = np.sum(np.abs(np_array)**2.0)*d_x*d_y/factor_t/factor_t*1e4
    
    I_defect = defect_power/norm
    
    I_defect_normalized = I_defect/defect_intensity_normalization
    
    return af.to_array(np.sqrt(I_defect_normalized)*np_array)

def droplet_2d_old(x_af, y_af, defect_power, defect_intensity_normalization, x0, y0, d1, d2, kz_times_z1, kz_times_z2, factor_t):
   
    np_array =  (jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d1 ))**1.0 
    np_array +=  (jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d2 ))**1.0 
    
    np_array = np.exp(1j*kz_times_z)
    
    x_max = np.max(x_af)/factor_t*1e3
    y_max = np.max(y_af)/factor_t*1e3
    
    d_x = (x_af[1,0]-x_af[0,0])
    d_y = (y_af[0,1]-y_af[0,0])
    
    d_x *= 5.0/x_max
    d_y *= 5.0/y_max
    
    norm = np.sum(np.abs(np_array)**2.0)*d_x*d_y/factor_t/factor_t*1e4
    
    I_defect = defect_power/norm
    
    I_defect_normalized = I_defect/defect_intensity_normalization
    
    return af.to_array(np.sqrt(I_defect_normalized)*np_array)


def add_field_from_array_with_velocity_normalized(array_to_add, x_af, y_af, Power_beam, intensity_normalization, 
                                                x0, y0, vx, vy, factor_t, reference_norm=0):
    
    
    
    d_x = x_af[1,0] - x_af[0,0]
    d_y = y_af[0,1] - y_af[0,0]
    

    
    current_norm = np.sum(np.abs(array_to_add)**2.0)*d_x*d_y/factor_t/factor_t*1e4
    
    if reference_norm==0:
        reference_norm = current_norm
    
    array_to_add = array_to_add*np.sqrt(reference_norm/current_norm)
    
    I_beam = Power_beam/reference_norm
    I_beam_normalized = I_beam/intensity_normalization
    
    new_field = af.interop.from_ndarray(np.sqrt(I_beam_normalized)*array_to_add)*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    
    return af.to_array(new_field)


def add_field_from_array_with_velocity(array_to_add, x_af, y_af, A, x0, y0, vx, vy):
    
    new_field = af.interop.from_ndarray(A*array_to_add)*af.exp(1.0j*(vx*(x_af-x0)+ vy*(y_af-y0)))
    
    return new_field

def droplet_2d(x_af, y_af, defect_power, defect_intensity_normalization, x0, y0, d1, d2, kz_times_z1, kz_times_z2, factor_t):
   
    np_array =  (jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d1 ))**1.0*np.exp(1j*kz_times_z1)
    np_array +=  (jn(0, 2.4048*(np.sqrt((x_af-x0)**2.0 + (y_af-y0)**2.0))/d2 ))**1.0*np.exp(1j*kz_times_z2)
    np_array = np_array.astype(np.complex128)
    
    x_max = np.max(x_af)/factor_t*1e3
    y_max = np.max(y_af)/factor_t*1e3
    
    d_x = (x_af[1,0]-x_af[0,0])
    d_y = (y_af[0,1]-y_af[0,0])
    
    d_x *= 5.0/x_max
    d_y *= 5.0/y_max
    
    norm = np.sum(np.abs(np_array)**2.0)*d_x*d_y/factor_t/factor_t*1e4
    
    I_defect = defect_power/norm
    
    I_defect_normalized = I_defect/defect_intensity_normalization
    
    return af.to_array(np.sqrt(I_defect_normalized)*np_array)
