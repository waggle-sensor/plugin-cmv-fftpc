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

from waggle.plugin import Plugin
from waggle.data.vision import Camera

import numpy as np
from scipy import stats as st
from inf import getInfoDict, cropMarginInfo, cropFrame, vectorMagnitudeDirection, getDivCurl
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

    

    with Plugin() as plugin, Camera(args.input) as camera:
        #get video frame and crop info into the dictionary.
        inf = cropMarginInfo(camera, inf)
    
        #Counting frames and time-steps for netcdf output requirment. 
        fcount = 0
        
        sample = camera.snapshot()
        frame_time = sample.timestamp
        fcount, sky_curr = cropFrame(sample, fcount, inf)

        run_on = True

        while run_on:
            if inf['interval'] > 0:
                time.sleep(inf['interval'])        
            #read new frame
            sample = camera.snapshot()
            frame_time = sample.timestamp
            fcount, sky_new = cropFrame(sample, fcount, inf)
            
            sky_prev = sky_curr
            sky_curr = sky_new
           
            #comput optical flow 
            with plugin.timeit("plg.inf.time_ns"):
                flow = cv2.calcOpticalFlowFarneback(sky_prev, sky_curr, None, 
                                                    pyr_scale=0.5, levels=3, 
                                                    winsize=inf['winsize'], 
                                                    iterations=3, poly_n=inf['poly_n'], 
                                                    poly_sigma=inf['poly_s'], flags=0)
                # Computes the magnitude and angle of the 2D vectors
                flow= np.round(flow, decimals=0)

                flow_u = np.ma.masked_equal(flow[..., 0], 0)
                flow_v = np.ma.masked_equal(flow[..., 1], 0)
                flow_u = np.ma.masked_where(np.ma.getmask(flow_v), flow_u)
                flow_v = np.ma.masked_where(np.ma.getmask(flow_u), flow_v)
                #magnitude, direction = cv2.cartToPolar(flow_u.mean(), flow_v.mean(), angleInDegrees=True)
                mag_mean, dir_mean = vectorMagnitudeDirection(flow_u.mean(),
                                                            flow_v.mean())
                mag_mode, dir_mode = vectorMagnitudeDirection(st.mode(flow_u),
                                                            st.mode(flow_v))
                mag_median, dir_median = vectorMagnitudeDirection(np.median(flow_u),
                                                            np.median(flow_v))
                div, curl = getDivCurl(flow)

                div = np.round(div.mean(), 2)
                curl = np.round(curl.mean(), 2)

    
            # Publish the output.an()
            plugin.publish('cmv.mean.vel', float(mag_mean))
            plugin.publish('cmv.mean.dir', float(dir_mean))
            plugin.publish('cmv.mode.vel', float(mag_mode))
            plugin.publish('cmv.mode.dir', float(dir_mode))
            plugin.publish('cmv.median.vel', float(mag_median))
            plugin.publish('cmv.median.dir', float(dir_median))

            #cv2.imwrite('/Users/bhupendra/image.jpg', sky_curr)
            #plugin.upload_file('/Users/bhupendra/image.jpg', meta={})
            #outfile.writelines(str(magnitude)+'\t'+str(direction)+'\n')
            

            #run_on = False
            
#outfile.close()



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
                        help='Time interval in seconds.', default=30)
    parser.add_argument('--c', type=int,
                        help='channels, 0=R, 1=G, 2=B, 9=Gr', default=0)
    parser.add_argument('--k', type=float,
                        help='Keep fraction of the image after cropping', default=0.9)
    parser.add_argument('--q', type=int, 
                        help='''Quality of the motion field.
                        Sets averaging window, poly_n and poly_sigma.
                        1-turbulant: detailed motion field but noisy.
                        2-smooth: lesser noise and fast computation,''',
                        default=1)

    
    args = parser.parse_args()
    main(args)