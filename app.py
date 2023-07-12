#!/usr/bin/env python3
"""
Created on Mon Nov 21 11:05:11 2022

Dense optical flow vectors are computed using OpenCV calcOpticalFlowFarneback method.
The method returns an array of 2-channel floating-point images that has the same size as 
imput images, and each pixel of the output array stores the computed optical flow for the 
corresponding pixel of the input images. The optical flow is represented as a 2D vector,
where the x and y components of the vector represent the flow along the x and y axis, 
respectively.

The plugin publishes the following data using the plugin.publish() method:
- 'cmv.thresh.otsu': Otsu threshold value for motion detection.
- 'cmv.motion.detected': Number of homogeneous motion segments.
- 'cmv.mean.mag.pxpm': Mean magnitude of motion in pixels per minute for each segment.
- 'cmv.mean.dir.degn': Mean direction of motion in degrees for each segment.
- 'cmv.median.mag.pxpm': Median magnitude of motion.
- 'cmv.median.dir.degn': Median direction of motion.

"""

import argparse
import sys
import time
import os
import logging

from waggle.plugin import Plugin
from waggle.data.vision import Camera

import numpy as np
from skimage.filters import threshold_otsu
from skimage import color
from skimage.segmentation import slic

from inf import getInfoDict, cropMarginInfo, cropFrame
import cv2



def upload_image(sky_curr, timestamp, thres_otsu, plugin):
    """
    Upload an image to beehive.

    This function uploads the sky_curr image, as a JPEG file.
    If an exception occurs during file removal, the error message is published.
    """
    img2_file_name = 'img2_' + str(timestamp) + '.jpg'
    cv2.imwrite(img2_file_name, sky_curr)
    plugin.upload_file(img2_file_name, meta={'thres_otsu': str(thres_otsu)}, timestamp=sample.timestamp)

    try:
        os.remove(img2_file_name)
    except Exception as e:
        plugin.publish('exit.error', e)
        sys.exit(-1)


def main(args):
    """ Runs the entire CMV workflow.
    
    Takes in input args and run the whole CMV workflow i.e. capture frames, 
    compute optical flow, perform clustering, and publish data to beehive.

    Args:
        args: The command-line arguments parsed by `argparse`.

    Returns:
        None
    """
    

    #Create a dictionary to save settings
    inf = getInfoDict(args)

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
            frame_time_curr = sample.timestamp/10**9
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
            frame_time_new = sample.timestamp/10**9
            fcount, sky_new = cropFrame(sample, fcount, inf)


            frame_time_prev = frame_time_curr
            frame_time_curr = frame_time_new

            vel_factor = 60/(frame_time_curr-frame_time_prev)

            sky_prev = sky_curr
            sky_curr = sky_new
           
            #comput optical flow 
#            with plugin.timeit("plg.inf.time_ns"):
            flow = cv2.calcOpticalFlowFarneback(sky_prev, sky_curr, None, 
                                                pyr_scale=0.5, levels=3, 
                                                winsize=inf['winsize'], 
                                                iterations=3, poly_n=inf['poly_n'], 
                                                poly_sigma=inf['poly_s'], flags=0)
            

            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = False)
            mag=mag * vel_factor
            #Use threshold for small values to remove background noise
            thres_otsu = np.round(threshold_otsu(mag))

            plugin.publish('cmv.thresh.otsu', float(thres_otsu), timestamp=sample.timestamp)

            # If it crossed the max threshold, upload sample image
            if thres_otsu > args.thr:
                upload_image(sky_curr, sample.timestamp, thres_otsu, plugin)
            
            thres_mag = thres_otsu
            #Reset the threshold for better range
            if thres_otsu < 2:
                thres_mag = 2
            if thres_otsu >10:
                thres_mag = 10
                


            mag_mask = np.repeat(mag[:, :, np.newaxis], 2, axis=2)

            flow= np.ma.masked_where(mag_mask<thres_mag, flow)

            if np.ma.MaskedArray.count(flow)==0:
                plugin.publish('cmv.motion.detected', int(-1), meta=meta, timestamp=sample.timestamp)
                continue


            #recompute the mag and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees = True)
            mag = mag * vel_factor
            ang = (90+ang) % 360


            # Number of Superpixels
            num_seg = args.segments

            #Use HSV coding for clustering
            hsv = np.zeros([mag.shape[0], mag.shape[1], 3], dtype=np.float32)
            hsv[..., 1] = 250 #255-curr_frame #250

            # Use Hue and Value to encode the Optical Flow
            hsv[..., 0] = ang/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # Convert HSV image to BGR
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # High compactness = squared regions & higher sigma = rounded segments
            # Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. 
            # (2012). SLIC superpixels compared to state-of-the-art superpixel methods.
            # IEEE transactions on pattern analysis and machine intelligence, 
            # 34(11), 2274-2282.
            segments = slic(image, n_segments=num_seg, sigma=2, compactness=10, convert2lab=True) 
            segments_found = segments.max()
            # Superpixel average color
            superpixels = color.label2rgb(segments, image, kind='avg')

            #first remove the zero vector region from segmentation
            seg_count = np.bincount(segments.ravel())

            motion_detected = 0

            for i in range(0, seg_count.shape[0]):
                seg_id = seg_count.argmax()
                seg_size = seg_count.max()
                mag_mean = np.mean(mag[segments==seg_id])
                ang_mean = np.mean(ang[segments==seg_id])
                mag_median = np.median(mag[segments==seg_id])
                ang_median = np.median(ang[segments==seg_id])
                
                # make it zero for next iteration
                seg_count[seg_id] = 0
                meta={'seg_id':str(seg_id),
                      'seg_size': str(seg_size),
                      'nsegments_found':str(segments_found),
                      'input': args.input,
                      'channel': str(args.c),
                      'image_frac': str(args.k),
                      'quality': str(args.q),
                      'nsegments_asked':str(args.segments)
                      }
                

                #Publish the output
                if not np.isnan(mag_mean) and float(mag_median) > thres_mag:
                    motion_detected = motion_detected + 1
                    meta['seg_rank']= str(motion_detected) # At this time `motions_detected` counter shows rank of the segment

                    plugin.publish('cmv.mean.mag.pxpm', np.round(mag_mean), meta=meta, timestamp=sample.timestamp)
                    plugin.publish('cmv.mean.dir.degn', np.round(ang_mean), meta=meta, timestamp=sample.timestamp)
                    plugin.publish('cmv.median.mag.pxpm', np.round(mag_median), meta=meta, timestamp=sample.timestamp)
                    plugin.publish('cmv.median.dir.degn', np.round(ang_median), meta=meta, timestamp=sample.timestamp)
                    #print('thres={} \t mag={} angle={}, seg_size={}, seg_id={}'.format(thres_mag, float(mag_mean), int(ang_mean), seg_size, seg_id))
            
            plugin.publish('cmv.motion.detected', motion_detected, meta=meta, timestamp=sample.timestamp)

                
            if args.oneshot:
                run_on = False
            




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
                                     This program uses optical flow method 
                                     from CV2 to compute the cloud 
                                     motion vectors in the hemispheric camera''')
    parser.add_argument('--input', type=str, 
                        help='Path to an input video or images.', 
                        default="file://test-data/sgp-sage.mp4")
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
                        default=50)
    parser.add_argument('--segments', type=int, 
                        help=''' Number of Segments. ''',
                        default=100)
    parser.add_argument('--oneshot', action= 'store_true',
                    help='''Run once and exit.''') #This is not working as intended when default option is used.

    
    args = parser.parse_args()
    main(args)
