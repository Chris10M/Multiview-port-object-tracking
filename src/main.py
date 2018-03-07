from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import threading

from video_device import VideoDevice

import time
import collections


class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)


class Constants:
    SAMPLE_TIME_IN_SECONDS = 5


class Events:
    frame_ready = threading.Event()
    ready_to_track = threading.Event()
    track_failure = threading.Event()


class Frame:
    file_name = 'test1.mp4'

    video = VideoDevice(device=file_name)
    video.start()

    @staticmethod
    def get():
        frame = Frame.video.get_frame()

        if frame is None:
            Events.frame_ready.clear()

        while frame is None:
            frame = Frame.video.get_frame()

        Events.frame_ready.set()

        return frame


class BoundHuman:
    """
    'groupThreshold – Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
     eps – Relative difference between sides of the rectangles to merge them into a group.'

    """
    sample_time = Constants.SAMPLE_TIME_IN_SECONDS

    winStride = (4, 4)
    padding = (8, 8)
    scale = 1.05
    overlapThresh = 0.65
    groupThreshold = 10
    eps = 0.2

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    @staticmethod
    def _sample_frame(frame_list):
        frame_list = np.array(frame_list)
        count = collections.defaultdict(int)

        for rects in frame_list:
            count[len(rects)] += 1

        count = [(k, count[k]) for k in sorted(count, key=count.get, reverse=True)]
        rect_length = count[0][0]
        frame_rects_list = []

        for j in range(rect_length):
            rects = []
            try:
                for i in range(len(frame_list)):
                    rects.append(frame_list[i][j].tolist())
            except IndexError:
                continue
            rects, _ = cv2.groupRectangles(rects,
                                           groupThreshold=BoundHuman.groupThreshold,
                                           eps=BoundHuman.eps)

            for rect in rects:
                frame_rects_list.append(rect)

        return frame_rects_list

    @staticmethod
    def get():
        current_time = time.time()
        while True:
            image = Frame.get()
            (rects, weights) = BoundHuman.hog.\
                                detectMultiScale(image,
                                                 winStride=BoundHuman.winStride,
                                                 padding=BoundHuman.padding,
                                                 scale=BoundHuman.scale)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

            pick = non_max_suppression(rects, probs=None, overlapThresh=BoundHuman.overlapThresh)

            if time.time() - current_time < BoundHuman.sample_time:
                pick_list.append(pick)
            else:
                break

        return BoundHuman._sample_frame(pick_list)




if __name__ == '__main__':

    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    pick_list = []

    while True:
        image = Frame.get()
        cv2.imshow("No EDIT", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    #current_time = time.time()

    #while True:
    '''
        
        frame = video.get_frame()

        if frame is None:
            continue

        image = frame

        # detect people in the image
        t = Benchmark()
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)

        #print("detectMultiScale fun fps: {0}".format(t.end()))

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

        t = Benchmark()
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        #print("non_max_suppression fun fps: {0}".format(t.end()))

        if time.time() - current_time < 5:
             pick_list.append(pick)
        else:
            break
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes]

        # show the output images
        cv2.imshow("After NMS", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''

    picked_list = BoundHuman.get()

    while True:
        frame = Frame.get()
        image = frame

        for (xA, yA, xB, yB) in picked_list:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cv2.imshow("No EDIT", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



    # When everything done, release the capture
    cv2.destroyAllWindows()