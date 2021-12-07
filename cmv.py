#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:48:13 2021
Parts of this module are adopted from TINT module phase_correlation.py
"""

import numpy as np
from scipy import ndimage


#for debugging 
#import matplotlib.pyplot as plt

def flowVectorSplit(array1, array2, info):
    """ Splits camera view into a grid, and computes flow vectors for each block.

    @ToDo: nblocks should be provided for both x and y direction to accomodate
    non-squared area, if needed.
    """
    nblock = info['nblock']
    
    array1_split = split2DArray(array1, nblock)
    array2_split = split2DArray(array2, nblock)
    
    cmv_x = np.zeros(nblock*nblock)
    cmv_y = np.zeros(nblock*nblock)
    for i in range(nblock*nblock):
        cmv_y[i], cmv_x[i] = fftFlowVector(array1_split[i], array2_split[i])
        
    cmv_x, cmv_y = rmLargeMagnitudes(cmv_x, cmv_y, v_max=info['v_max'])
    
    cmv_x = cmv_x.reshape([nblock, nblock])
    cmv_y = cmv_y.reshape([nblock, nblock])
    
    cmv_x, cmv_y, nmf_u, nmf_v = rmSpuriousVectors(cmv_x, cmv_y, info)
    
    return cmv_x, cmv_y


def rmLargeMagnitudes(cmv_x, cmv_y, v_max):
    """
    Remove large anomalous values. Not using direction.

    @ToDo: Trim the cmv (not remove) for large magnitude by the ratio 
    vmag/v_max.
    """
    
    vmag, vdir = vectorMagnitudeDirection(cmv_x, cmv_y) 
    cmv_x[np.where(abs(vmag)>v_max)]=0
    cmv_y[np.where(abs(vmag)>v_max)]=0
    
    return cmv_x, cmv_y



def rmSpuriousVectors(cmv_x, cmv_y, info):
    """ Remove large anomalous values.
    """
    norm_fluct_u, norm_fluct_v  = getNormMedianFluctuation(cmv_x, cmv_y, info)
    norm_fluct_mag = np.sqrt(norm_fluct_u**2 + norm_fluct_v**2)
    error_flag = norm_fluct_mag > info['WS05-error_thres']
 
    cmv_x[np.where(error_flag)] = 0
    cmv_y[np.where(error_flag)] = 0
    
    return cmv_x, cmv_y, norm_fluct_u, norm_fluct_v


def getNormMedianFluctuation(u, v, info):
    """ Checking vector validity using normaliazed median fluctuatin method 
    from Westerweel and Scarano (2005).
    
    
    Westerweel, J., & Scarano, F. (2005). Universal outlier detection for PIV
    data. Experiments in fluids, 39(6), 1096-1100.
    """
    d = info['WS05-neighborhood_dist']
    eps = info['WS05-eps'] 
    
    norm_fluctuation_u = normFluctuation(u, d, eps)
    norm_fluctuation_v = normFluctuation(v, d, eps)
    
    return norm_fluctuation_u, norm_fluctuation_v
    


def normFluctuation(vel_comp, d, eps): 
    v_shape = vel_comp.shape   
    norm_fluctuation = np.zeros(v_shape)
    norm_fluctuation[:] = np.NaN
    
    for i in range(d, v_shape[0]-d):
        for j in range(d, v_shape[1]-d):
            neighborhood = vel_comp[i-d:i+d+1, j-d:j+d+1]
            
            #remove central point
            neighborhood = neighborhood.flatten()
            mid_point = int(np.floor(neighborhood.size/2))
            neighborhood = np.delete(neighborhood, mid_point)
            
            neigh_median = np.median(neighborhood)
            
            fluctuation = vel_comp[i, j]-neigh_median
            
            residue = neighborhood-neigh_median
            residue_median = np.median(np.abs(residue))

            norm_fluctuation[i, j] = fluctuation/(residue_median+eps)
    
    norm_fluctuation=np.nan_to_num(norm_fluctuation)
    return norm_fluctuation

    


def vectorMagnitudeDirection(cmv_x, cmv_y, std_fact=1):
    vec_mag = np.sqrt(cmv_x*cmv_x + cmv_y*cmv_y)
    vec_dir = np.rad2deg(np.arctan2(cmv_y,cmv_x)) % 360
    
    return vec_mag, vec_dir


def fftFlowVector(im1, im2, global_shift=True):
    """ Estimates flow vectors in two images using cross covariance. """
    if not global_shift and (np.max(im1) == 0 or np.max(im2) == 0):
        return None

    crosscov = fftCrossCov(im1, im2)
    sigma = 3
    cov_smooth = ndimage.filters.gaussian_filter(crosscov, sigma)
    dims = np.array(im1.shape)

    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    
    rs = np.ceil(dims[0]/2).astype('int')
    cs = np.ceil(dims[1]/2).astype('int')

    # Calculate shift relative to center - see fft_shift.
    pshift = pshift - (dims - [rs, cs])
    return pshift


def fftCrossCov(im1, im2):
    """ Computes cross correlation matrix using FFT method. """
    fft1_conj = np.conj(np.fft.fft2(im1))
    fft2 = np.fft.fft2(im2)
    normalize = abs(fft2 * fft1_conj)
    try:  min_value = normalize[(normalize > 0)].min()
    except ValueError:  #raised if empty.
       min_value=0.01
       pass  
    normalize[normalize == 0] = min_value  # prevent divide by zero error
    cross_power_spectrum = (fft2 * fft1_conj)/normalize
    crosscov = np.fft.ifft2(cross_power_spectrum)
    crosscov = np.real(crosscov)
    return motionVector(crosscov)


def motionVector(fft_mat):
    """ Rearranges the cross correlation matrix so that 'zero' frequency or DC
    component is in the middle of the matrix. Taken from stackoverflow Que.
    30630632.
    """
    if type(fft_mat) is np.ndarray:
        rs = np.ceil(fft_mat.shape[0]/2).astype('int')
        cs = np.ceil(fft_mat.shape[1]/2).astype('int')
        quad1 = fft_mat[:rs, :cs]
        quad2 = fft_mat[:rs, cs:]
        quad3 = fft_mat[rs:, cs:]
        quad4 = fft_mat[rs:, :cs]
        centered_t = np.concatenate((quad4, quad1), axis=0)
        centered_b = np.concatenate((quad3, quad2), axis=0)
        centered = np.concatenate((centered_b, centered_t), axis=1)
        # Thus center is formed by shifting the entries of fft_mat
        # up/left by [rs, cs] indices, or equivalently down/right by
        # (fft_mat.shape - [rs, cs]) indices, with edges wrapping. 
        return centered
    else:
        print('input to motionVector() should be a matrix')
        return


def split2DArray(arr2d, nblock):
    """ Splits sky into given number of blocks. 
 
    Not tested for uneven shapes or nonfitting arrays """
    split_0 = np.array_split(arr2d, nblock, axis=0)
    split_arr=[]
    for arr in split_0:
        split_01 = np.array_split(arr, nblock, axis=1)
        split_arr += split_01
    
    return split_arr



def meanCMV(u, v):
    """ Computes mean field CMV.
    """
    if(np.all(u==0)):
        u_mean=0
    else:
        u_mean = u[(np.abs(u) > 0) | (np.abs(v) > 0)].mean()
        
    if(np.all(v==0)):
        v_mean=0
    else:
        v_mean = v[(np.abs(u) > 0) | (np.abs(v) > 0)].mean()
        
    return u_mean, v_mean
