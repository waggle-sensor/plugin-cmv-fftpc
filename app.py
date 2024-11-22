#!/usr/bin/env python3
"""
Created on Mon Nov 21 11:05:11 2022

Dense optical flow vectors are computed using OpenCV calcOpticalFlowFarneback method.
...
"""

import argparse
import sys
import time
import logging
from functools import partial

import numpy as np
from skimage.filters import threshold_otsu
from skimage import color
from skimage.segmentation import slic

from waggle.plugin import Plugin
from waggle.data.vision import Camera
from inf import getInfoDict, cropMarginInfo, cropFrame
import cv2

# Configure logger
logging.basicConfig()
logger = logging.getLogger(__name__)

def upload_image(sky_curr, timestamp, thres_otsu, plugin):
    img2_file_name = f"img2_{timestamp}.jpg"
    cv2.imwrite(img2_file_name, sky_curr)
    plugin.upload_file(
        img2_file_name, meta={"thres_otsu": str(thres_otsu)}, timestamp=timestamp
    )


def initialize_plugin(args):
    inf = getInfoDict(args)
    with Plugin() as plugin:
        with Camera(args.input) as camera:
            inf = cropMarginInfo(camera, inf)
    return inf


def process_frame(camera, inf, fcount):
    sample = camera.snapshot()
    frame_time = sample.timestamp / 10**9
    fcount, frame = cropFrame(sample, fcount, inf)
    return frame_time, frame, fcount


def compute_optical_flow(sky_prev, sky_curr, inf, vel_factor):
    flow = cv2.calcOpticalFlowFarneback(
        sky_prev,
        sky_curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=inf["winsize"],
        iterations=3,
        poly_n=inf["poly_n"],
        poly_sigma=inf["poly_s"],
        flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    mag *= vel_factor
    ang = (90 + ang) % 360
    return flow, mag, ang


def threshold_and_publish(mag, plugin, sample, thres_threshold):
    thres_otsu = np.round(threshold_otsu(mag))
    plugin.publish("cmv.thresh.otsu", float(thres_otsu), timestamp=sample.timestamp)
    return thres_otsu > thres_threshold, thres_otsu


def create_segments(image, mag, ang, num_seg):
    hsv = np.zeros([mag.shape[0], mag.shape[1], 3], dtype=np.float32)
    hsv[..., 1] = 250
    hsv[..., 0] = ang / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    segments = slic(image, n_segments=num_seg, sigma=2, compactness=10, convert2lab=True)
    return segments, color.label2rgb(segments, image, kind="avg")


def analyze_and_publish_segments(segments, mag, ang, plugin, sample, args):
    seg_count = np.bincount(segments.ravel())
    motion_detected = 0

    for _ in range(args.seg_pub):
        seg_id = seg_count.argmax()
        seg_size = seg_count.max()
        mag_mean = np.mean(mag[segments == seg_id])
        ang_mean = np.mean(ang[segments == seg_id])
        mag_median = np.median(mag[segments == seg_id])
        ang_median = np.median(ang[segments == seg_id])

        seg_count[seg_id] = 0
        motion_detected += 1
        meta = {
            "seg_id": str(seg_id),
            "seg_size": str(seg_size),
            "nsegments_found": str(segments.max()),
            "input": args.input,
            "channel": str(args.c),
            "image_frac": str(args.k),
            "quality": str(args.q),
            "nsegments_asked": str(args.segments),
            "seg_rank": str(motion_detected),
        }

        plugin.publish("cmv.mean.mag.pxpm", float(np.round(mag_mean)), meta=meta, timestamp=sample.timestamp)
        plugin.publish("cmv.mean.dir.degn", float(np.round(ang_mean)), meta=meta, timestamp=sample.timestamp)
        plugin.publish("cmv.median.mag.pxpm", float(np.round(mag_median)), meta=meta, timestamp=sample.timestamp)
        plugin.publish("cmv.median.dir.degn", float(np.round(ang_median)), meta=meta, timestamp=sample.timestamp)

    plugin.publish("cmv.motion.detected", motion_detected, timestamp=sample.timestamp)


def main(args):
    inf = initialize_plugin(args)
    fcount = 0
    with Plugin() as plugin, Camera(args.input) as camera:
        frame_time_curr, sky_curr, fcount = process_frame(camera, inf, fcount)

        logger.info("Starting main loop")
        while True:
            logger.debug("Inside main loop")
            logger.debug("Processing frame")
            if inf["interval"] > 0:
                time.sleep(inf["interval"])

            with Camera(args.input) as camera:
                frame_time_prev = frame_time_curr
                frame_time_curr, sky_new, fcount = process_frame(camera, inf, fcount)

            logger.debug("Computing optical flow")
            vel_factor = 60 / (frame_time_curr - frame_time_prev)
            sky_prev, sky_curr = sky_curr, sky_new

            flow, mag, ang = compute_optical_flow(sky_prev, sky_curr, inf, vel_factor)
            logger.debug("Thresholding and publishing")
            should_upload, thres_otsu = threshold_and_publish(mag, plugin, camera.snapshot(), args.thr)

            if should_upload:
                upload_image(sky_curr, camera.snapshot().timestamp, thres_otsu, plugin)

            logger.debug("Creating segments")
            segments, _ = create_segments(sky_curr, mag, ang, args.segments)
            logger.debug("Analyzing and publishing segments")
            analyze_and_publish_segments(segments, mag, ang, plugin, camera.snapshot(), args)

            if args.oneshot:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optical flow motion detection.")
    parser.add_argument("--input", type=str, default="file://test-data/sgp-sage.mp4")
    parser.add_argument("--i", type=int, default=30)
    parser.add_argument("--c", type=int, default=0)
    parser.add_argument("--k", type=float, default=0.9)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--thr", type=int, default=50)
    parser.add_argument("--segments", type=int, default=100)
    parser.add_argument("--seg_pub", type=int, default=9)
    parser.add_argument("--oneshot", action="store_true")
    args = parser.parse_args()
    main(args)
