#!/usr/bin/env python3
"""
Created on Mon Oct 25 11:05:11 2021

The sky is divided into the kxk square blocks of block-size lxl pixels.
"""
import argparse
import sys
import time

from waggle.plugin import Plugin
from waggle.data.vision import Camera


from inf import getInfoDict, cropMarginInfo, cropFrame
from cmv import flowVectorSplit, meanCMV

#For debugging
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation


def main(args):
    """ Takes in input args and run the whole CMV workflow.
    """
    
    #Create a dictionary to save settings
    inf = getInfoDict(args, NDist=1, error_thres=6, eps=0.2)

    with Plugin() as plugin, Camera(args.input) as camera:
        #get video frame and crop info into the dictionary.
        inf = cropMarginInfo(camera, inf)
    
        #Counting frames and time-steps for netcdf output requirment. 
        fcount = 0
        first_frame = True
        oneshot = True
        
        while oneshot:
            sample = camera.snapshot()
            frame_time = sample.timestamp
            fcount, sky_new = cropFrame(sample, fcount, inf)
    
            #Store the sky data for first the frame and and wait for the next frame.
            if first_frame:
                sky_curr = sky_new
                first_frame = False
                if inf['interval'] > 0:
                    time.sleep(inf['interval'])
                continue
        
            
            #move one frame forward
            sky_prev = sky_curr
            sky_curr = sky_new
            
            #Split the image and comput flow for all image blocks
            start_time = time.time_ns()
            cmv_x, cmv_y = flowVectorSplit(sky_prev, sky_curr, inf)
            u_mean,  v_mean = meanCMV(cmv_x, cmv_y)
            end_time = time.time_ns()
            inference_time = end_time-start_time
    
            # Publish the output.
            plugin.publish('atm.cmv.mean.u', u_mean)
            plugin.publish('atm.cmv.mean.v', v_mean)
            plugin.publish('atm.cmv.time', frame_time)
            plugin.publish('plg.inf.time_ns', inference_time)
            #ugin.upload_file()
            
            oneshot = False
            #if inf['interval'] > 0:
            #    time.sleep(inf['interval'])
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                     This program uses phase correlation method 
                                     from the TINT module to compute the cloud 
                                     motion vectors in the hemispheric camera''')
    parser.add_argument('--input', type=str, 
                        help='Path to an input video or images.', 
                        default="file://test-data/sgptsimovieS01.a1.20160726.000000.mpg")
#                        default="/app/test-data/sgptsimovieS01.a1.20160726.000000.mpg")   
    parser.add_argument('--i', type=int, 
                        help='Time skip in seconds.', default=30)
    parser.add_argument('--k', type=int, 
                        help='kxk image sectors used for CMV computation.',
                        default=10)
    parser.add_argument('--l', type=int,
                        help='square block length, lxl in pixels.', default=40)
    parser.add_argument('--c', type=int,
                        help='RGB channels, 0=R, 1=G, 2=B', default=0)
    
    args = parser.parse_args()
    main(args)










