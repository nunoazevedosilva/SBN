# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:16:29 2022

@author: NonlinearOpticalLab
"""

import numpy as np

import scipy as sp
import scipy.fftpack as sfft
from scipy.ndimage import shift
from scipy import interpolate
from scipy.special import jv, jn
from scipy.fftpack import fft, fft2, fftfreq, fftshift, fftn, ifft2, convolve

def poisson_eq(ux, uy, dx, dy):
    fftx = fft2(ux)
    ffty = fft2(uy)
    fx = fftfreq(len(ux[:, 0]), dx) * 2 * np.pi
    fy = fftfreq(len(ux[0, :]), dy) * 2 * np.pi

    kx, ky = np.meshgrid(fx, fy, indexing='ij')

    auxiliar = 1.0j * (kx * ux + ky * uy) / (kx ** 2 + ky ** 2 + 0.0001)
    theta = ifft2(auxiliar)

    return theta, kx, ky

def get_field_turbulence_properties_single_field(field1, dx, dy, k_n=256):


    ux1 = np.imag(np.conj(field1) * (np.roll(field1, 1, 0) - np.roll(field1, -1, 0)) / (
                2 * dx)) / np.abs(field1)
    uy1 = np.imag(np.conj(field1) * (np.roll(field1, 1, 1) - np.roll(field1, -1, 1)) / (
                2 * dy)) / np.abs(field1)


    theta, kx, ky = poisson_eq(ux1, uy1, dx, dy)

    fux1 = fft2(ux1)
    fuy1 = fft2(uy1)

    kx = np.array(kx)
    ky = np.array(ky)

    f_uix1 = fux1 - kx * (kx * fux1 + ky * fuy1) / (kx * kx + ky * ky + 1e-12)
    f_uiy1 = fuy1 - ky * (kx * fux1 + ky * fuy1) / (kx * kx + ky * ky + 1e-12)


    uix1 = ifft2(f_uix1)
    uiy1 = ifft2(f_uiy1)


    f_ucx1 = (kx * kx * fux1 + kx * ky * fuy1) / (kx * kx + ky * ky + 0.000000001)
    f_ucy1 = (ky * kx * fux1 + ky * ky * fuy1) / (kx * kx + ky * ky + 0.000000001)

    ucx1 = ifft2(f_ucx1)
    ucy1 = ifft2(f_ucy1)
   

    kmax = np.sqrt(kx ** 2 + ky ** 2).max()
    kmin = np.sqrt(kx ** 2 + ky ** 2).min()
    
    kks = np.array(fftfreq(ucx1.shape[0], dx)) * 2 * np.pi

    ecik = np.zeros(len(kks))
    ecck = np.zeros(len(kks))

    R = np.sqrt(kx ** 2 + ky ** 2)
    image = (f_ucx1 * np.conj(f_ucx1) + f_ucy1 * np.conj(f_ucy1))
    image2 = (f_uix1 * np.conj(f_uix1) + f_uiy1 * np.conj(f_uiy1))
    r = np.linspace(kmin, kmax, k_n)

    dr = r[1] - r[0]


    ecc = lambda r: (image[(R >= r - dr / 2.) & (R < r + dr / 2.)].mean())
    eci = lambda r: (image2[(R >= r - dr / 2.) & (R < r + dr / 2.)].mean())


    mean_eci = (np.vectorize(eci)(r)) * (2 * np.pi * r)
    mean_ecc = (np.vectorize(ecc)(r)) * (2 * np.pi * r)

    
    return r, mean_eci, mean_ecc

def derivative_2_order_2d(field, dh_spacing, axis_value=1):
    result = -(np.roll(field, -2, axis_value) - np.roll(field, 2, axis_value)) / (12.0 * dh_spacing) + \
             2.0 * (np.roll(field, -1, axis_value) - np.roll(field, 1, axis_value)) / (3.0 * dh_spacing)

    return result

def get_vorticity(field1, dx, dy):

    
    ux1 = np.imag(np.conj(field1)*derivative_2_order_2d(field1, dx, 1)) / np.abs(field1)
               
    uy1 = np.imag(np.conj(field1)*derivative_2_order_2d(field1, dy, 0)) / np.abs(field1)
    
    w1 = derivative_2_order_2d(uy1, dx, 1) - derivative_2_order_2d(ux1, dy, 0)

    
    return w1

def get_momentum_vector_2d(field, dx, dy):
   
    px_i = np.conjugate(field)*derivative_2_order_2d(field, dx, 1)
    py_i = np.conjugate(field)*derivative_2_order_2d(field, dy, 0)
    return np.imag(px_i), np.imag(py_i)