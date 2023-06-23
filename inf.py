#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:06:48 2021

"""
import sys
import numpy as np
import cv2



def mask_threshold_flow(flow, threshold=1):
    magnitude, _ = vectorMagnitudeDirection(flow)
    mask = magnitude > threshold
    return flow[np.logical_not(mask)]


def getDivCurl(flow):
    flow_div = divergence_npgrad(flow)
    flow_curl = curl_npgrad(flow)
    return flow_div, flow_curl

def divergence_npgrad(flow):
    """ This function is borrowed from SO

    https://stackoverflow.com/a/71992362/1227454
    """
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dx = np.gradient(Fx, axis=0)
    dFy_dy = np.gradient(Fy, axis=1)
    return dFx_dx + dFy_dy

def curl_npgrad(flow):
    """ This function is borrowed from SO

    https://stackoverflow.com/a/71992362/1227454
    """
    flow = np.swapaxes(flow, 0, 1)
    Fx, Fy = flow[:, :, 0], flow[:, :, 1]
    dFx_dy = np.gradient(Fx, axis=1)
    dFy_dx = np.gradient(Fy, axis=0)
    curl = dFy_dx - dFx_dy
    return curl


def getInfoDict(args):
    """ Takes input args and save it in a dictionary.
    """
    inf = dict()
    inf['input'] = args.input
    inf['interval'] = args.i
    inf['channel'] = args.c
    inf['keep_frac'] = args.k
    inf['quality'] = args.q #resets to computed values if zero
    inf = set_quality(inf)
    return inf


def cropMarginInfo(camera, inf):
    """
    Returns crop boundaries, block locations and frame info in a dictionary. 
    """
    frame = camera.snapshot()
    frame_height = frame.data.shape[0]
    frame_width = frame.data.shape[1]
    inf['frame_height'] = frame_height
    inf['frame_width'] = frame_width
    
    inf = getCropMargin(inf)
    return inf


def set_quality(inf):
    '''Compute the averaging window size for optical flow if not provided.'''
    
    if inf['quality'] == 1:
         inf['winsize'] = 15
         inf['poly_n'] = 5
         inf['poly_s'] = 1.2
         
    elif inf['quality'] == 2:
         inf['winsize'] = 20
         inf['poly_n'] = 7
         inf['poly_s'] = 1.5        
    return inf



def getCropMargin(inf):
    """ Computes crop area from the settings from 'inf' and
    appends it to the same dictionary. 
    """
    
    small_dim_len = min(inf['frame_height'], inf['frame_width'])
    
    inf['crop_len'] = small_dim_len *  inf['keep_frac']
    
    
    if(inf['crop_len'] > small_dim_len):
        exit("Unexpected Error: The original frame size is smaller than \
             the provided crop size. check k<1")
             
    inf['cent_x'] = int(inf['frame_width']/2)
    inf['cent_y'] = int(inf['frame_height']/2)
    
    #crop a square region of interest to accomodate 
    inf['y1'] = int(inf['cent_y'] - inf['crop_len'] /2)
    inf['y2'] = int(inf['cent_y'] + inf['crop_len'] /2)
    inf['x1'] = int(inf['cent_x'] - inf['crop_len'] /2)
    inf['x2'] = int(inf['cent_x'] + inf['crop_len'] /2)
    
    return inf

def cropFrame(sample, fcount, inf):
    """ Extract and crops frame using crop margins from the 'inf'.
    """
    lap_kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    
    frame = sample.data
    sky = frame[inf['y1']:inf['y2'], inf['x1']:inf['x2'], :]
    if inf['channel'] == 9:
        sky = cv2.cvtColor(sky, cv2.COLOR_RGB2GRAY)
    else:
        sky = sky[:, :, inf['channel']]
    fcount += 1
    
    #sys.stdout.write('Current Frame:' + str(fcount)+ '\r')
    #sys.stdout.flush()

    sky = cv2.filter2D(sky, -1, lap_kernel)

    return fcount, sky


def vectorMagnitudeDirection(cmv_x, cmv_y):
    """ The function calculates the magnitude (length) 
    and direction of the vector, and returns both values as a tuple.
    """
    vec_mag = np.sqrt(cmv_x*cmv_x + cmv_y*cmv_y)
    vec_dir = (90-np.rad2deg(np.arctan2(cmv_y,cmv_x))) % 360
    
    return vec_mag, vec_dir




def floorSmallMagnitudes(cmv_x, cmv_y):
    """
    Remove small values. Not using direction.

    @ToDo: Trim the cmv (not remove) for large magnitude by the ratio 
    vmag/v_max.
    """
    
    vmag, vdir = vectorMagnitudeDirection(cmv_x, cmv_y) 
    cmv_x[np.where(abs(vmag)<1)]=0
    cmv_y[np.where(abs(vmag)<1)]=0
    
    return cmv_x, cmv_y

