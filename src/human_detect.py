import cv2
import collections
import time
import numpy as np
from imutils.object_detection import non_max_suppression

from video_device import Frame


class Constant:
    SAMPLE_TIME_IN_SECONDS = 5


class BoundHuman:
    """
     groupThreshold – Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
     eps – Relative difference between sides of the rectangles to merge them into a group.

    """
    sample_time = Constant.SAMPLE_TIME_IN_SECONDS

    winStride = (4, 4)
    padding = (8, 8)
    scale = 1.05
    overlapThresh = 0.65

    groupThreshold = 20
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
        start_time = time.time()
        pick_list = list()
        while True:
            image = Frame.get()
            (rects, weights) = BoundHuman.hog. \
                detectMultiScale(image,
                                 winStride=BoundHuman.winStride,
                                 padding=BoundHuman.padding,
                                 scale=BoundHuman.scale)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

            pick = non_max_suppression(rects,
                                       probs=None,
                                       overlapThresh=BoundHuman.overlapThresh)

            if time.time() - start_time < BoundHuman.sample_time:
                pick_list.append(pick)
            else:
                break

        picked_list = BoundHuman._sample_frame(pick_list)

        return picked_list
