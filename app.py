#!/usr/bin/env python3
"""
Created on Mon Nov 21 11:05:11 2022

Dense optical flow vectors are computed using OpenCV calcOpticalFlowFarneback method.
The method returns an array of 2-channel floating-point images that has the same size as 
imput images, and each pixel of the output array stores the computed optical flow for the 
corresponding pixel of the input images. The optical flow is represented as a 2D vector,
where the x and y components of the vector represent the flow along the x and y axis, 
respectively.
"""

import argparse
import sys
import time
import os
import logging

from waggle.plugin import Plugin
from waggle.data.vision import Camera

import numpy as np
#from scipy import stats as st
from inf import getInfoDict, cropMarginInfo, cropFrame, vectorMagnitudeDirection
import cv2



###For debugging
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

#txt_file = '/Users/bhupendra/Desktop/oftest_flip.txt'

#outfile = open(txt_file, "a")
#outfile.writelines('magnitude\tdirection\n')

def main(args):
    """ Takes in input args and run the whole CMV workflow.
    """
    
    #Create a dictionary to save settings
    inf = getInfoDict(args)

    vel_factor = 60/inf['interval']

    with Plugin() as plugin:
        #get video frame and crop info into the dictionary.
        with Camera(args.input) as camera:
            inf = cropMarginInfo(camera, inf)
    
        #Counting frames and time-steps for netcdf output requirment. 
        fcount = 0
        
        with Camera(args.input) as camera:
            try:
                sample = camera.snapshot()
            except Exception as e:
                logging.error(f"Run-time error in First camera.snapshot: {e}")
                plugin.publish('exit.error', e)
                sys.exit(-1)
            frame_time = sample.timestamp
            fcount, sky_curr = cropFrame(sample, fcount, inf)

        run_on = True

        while run_on:
            if inf['interval'] > 0:
                time.sleep(inf['interval'])        
            #read new frame
            with Camera(args.input) as camera:
                try:
                    sample = camera.snapshot()
                except Exception as e:
                    logging.error(f"Run-time error in Second camera.snapshot: {e}")
                    plugin.publish('exit.error', e)
                    sys.exit(-1)
            frame_time = sample.timestamp
            fcount, sky_new = cropFrame(sample, fcount, inf)
            
            sky_prev = sky_curr
            sky_curr = sky_new
           
            #comput optical flow 
#            with plugin.timeit("plg.inf.time_ns"):
            flow = cv2.calcOpticalFlowFarneback(sky_prev, sky_curr, None, 
                                                pyr_scale=0.5, levels=3, 
                                                winsize=inf['winsize'], 
                                                iterations=3, poly_n=inf['poly_n'], 
                                                poly_sigma=inf['poly_s'], flags=0)
            
            # Computes the magnitude and angle of the 2D vectors
            flow= np.floor(flow)

            flow_u = np.ma.masked_equal(flow[..., 0], 0)
            flow_v = np.ma.masked_equal(flow[..., 1], 0)
            mask = np.ma.mask_or(np.ma.getmask(flow_v), np.ma.getmask(flow_u))
            flow_u = np.ma.masked_where(mask, flow_u)
            flow_v = np.ma.masked_where(mask, flow_v)

            mag_mean, dir_mean = vectorMagnitudeDirection(flow_u.mean(),
                                                        flow_v.mean())
            mag_mean_minute = np.round(mag_mean * vel_factor, decimals=0)
            
            #mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])


    
            # Publish the output
            plugin.publish('cmv.mean.vel.pixpmin', float(mag_mean_minute), timestamp=frame_time)
            plugin.publish('cmv.mean.dir.degrees', float(dir_mean), timestamp=frame_time)
            #plugin.publish('cmv.mean.u.debug', float(flow_u.mean()))
            #plugin.publish('cmv.mean.v.debug', float(flow_v.mean()))

            


            # If it crossed the threshold, upload both images
            if mag_mean_minute > args.thr:
                img2_file_name = 'img2_'+str(frame_time)+'.jpg'
                img1_file_name = 'img1_'+str(frame_time)+'.jpg'
                cv2.imwrite(img2_file_name, sky_curr)
                cv2.imwrite(img1_file_name, sky_prev)
                plugin.upload_file(img2_file_name, meta={})
                plugin.upload_file(img1_file_name, meta={})
                try:
                    os.remove(img2_file_name)
                    os.remove(img1_file_name)
                except: pass
                
            if args.oneshot:
                run_on = False
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                     This program uses optical flow method 
                                     from CV2 to compute the cloud 
                                     motion vectors in the hemispheric camera''')
    parser.add_argument('--input', type=str, 
                        help='Path to an input video or images.', 
                        default="file://test-data/sgptsimovieS01.a1.20160726.000000.mpg")
#                        default="/app/test-data/sgptsimovieS01.a1.20160726.000000.mpg")   
    parser.add_argument('--i', type=int, 
                        help='Time interval in seconds.', default=30)
    parser.add_argument('--c', type=int,
                        help='channels, 0=R, 1=G, 2=B, 9=Gr', default=0)
    parser.add_argument('--k', type=float,
                        help='Keep fraction of the image after cropping', default=0.9)
    parser.add_argument('--q', type=int, 
                        help='''Quality of the motion field.
                        Sets averaging window, poly_n and poly_sigma.
                        1-turbulant: detailed motion field but noisy.
                        2-smooth: less noisy and faster computation,''',
                        default=2)
    parser.add_argument('--thr', type=int, 
                        help='''Uploads images when magnitude is above this threshold''',
                        default=10)
    parser.add_argument('--oneshot', action= 'store_true',
                    help='''Run once and exit.''') #This is not working as intended when default option is used.

    
    args = parser.parse_args()
    main(args)
