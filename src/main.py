from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import threading

from video_device import VideoDevice
#from tracker import Tracker

import time
import collections
import queue

class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)


class Constant:
    SAMPLE_TIME_IN_SECONDS = 5
    TRACKING_FAILURE_DETECTED = -1
    TIME_OUT_SECOND_FOR_TRACK = 1
    QUEUE_BUFFER_SIZE = 50


class Event:
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
            Event.frame_ready.clear()

        while frame is None:
            frame = Frame.video.get_frame()

        Event.frame_ready.set()

        return frame

class Buffer:
    buffer_size = Constant.QUEUE_BUFFER_SIZE

    def __init__(self):
        self.buffer_queue = queue.Queue(Buffer.buffer_size)

    def put(self, item):
        try:
            self.buffer_queue.put_nowait(item)

        except queue.Full:
            self.buffer_queue.get_nowait()
            self.buffer_queue.task_done()

            self.buffer_queue.put_nowait(item)

    def get_all(self):
        items = list()

        while True:
            try:
                item = self.buffer_queue.get_nowait()
                items.append(item)

            except queue.Empty:
                return items


class Tracker(threading.Thread):
    time_out_seconds = Constant.TIME_OUT_SECOND_FOR_TRACK
    g_id = 0
    g_id_lock = threading.Lock()

    POISON_PILL = (0, 0, 0, 0)

    def __init__(self, bounding_box, _id=None):
        threading.Thread.__init__(self)
        self.tracker = cv2.TrackerKCF_create()
        self.bounding_box_queue = queue.Queue()
        _ = self.tracker.init(Frame.get(), tuple(bounding_box))
        self.id_lock = threading.Lock()

        self.buffer_queue = Buffer()

        if _id is not None:
            with self.id_lock:
                self.id = _id
        else:
            with Tracker.g_id_lock:
                with self.id_lock:
                    self.id = Tracker.g_id
                Tracker.g_id += 1

    def run(self):
        track_time = time.time()
        while True:
            frame = Frame.get()
            ok, bbox = self.tracker.update(frame)

            if ok:
                self.bounding_box_queue.put(bbox)

                self.buffer_queue.put(frame[int(bbox[1]):\
                                            int(bbox[3]),\
                                            int(bbox[0]):\
                                            int(bbox[2])])

                track_time = time.time()
            else:
                Event.track_failure.set()
                self.bounding_box_queue.put(Tracker.POISON_PILL)

            if Event.track_failure.is_set():
                if time.time() - track_time > Tracker.time_out_seconds:
                    break

    def get_bounding_box(self, dtype=float):
        bounding_box = self.bounding_box_queue.get()
        self.bounding_box_queue.task_done()

        if dtype == int:
            return int(bounding_box[0]),\
                   int(bounding_box[1]),\
                   int(bounding_box[2]),\
                   int(bounding_box[3])

        return bounding_box

    def get_id(self):
        with self.id_lock:
            return self.id

    def get_buffer(self):
        return self.buffer_queue.get_all()

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
        while True:
            image = Frame.get()
            (rects, weights) = BoundHuman.hog.\
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


if __name__ == '__main__':
    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


    pick_list = []

    ctime = time.time()

    while True:
        image = Frame.get()
        cv2.imshow("No EDIT", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - ctime > 1:
            break
    cv2.destroyAllWindows()

    '''
    current_time = time.time()
    
    while True:

        
        frame = Frame.get()

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

    picked_list = BoundHuman._sample_frame(pick_list)
    '''
    picked_list = BoundHuman.get()

    print(picked_list)

    roi_track_list = list()

    for roi in picked_list:
        thread = Tracker(roi)
        thread.daemon = True
        thread.start()
        roi_track_list.append(thread)

##       print(roi_box)

    while True:
        frame = Frame.get()
        image = frame
        for tracked_roi in roi_track_list:
            if Event.track_failure.is_set():
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)
                l = (tracked_roi.get_buffer())

                for i in l:
                    print(np.shape(i))
            else:
            #print(tracked_roi.get_bounding_box())
                b = Benchmark()
                xa, ya, xb, yb = tracked_roi.get_bounding_box(dtype=int)
                0,    1, 2,  3
                print(b.end())
                print(tracked_roi.get_id())
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 0, 255 * tracked_roi.get_id()), 2)
            #time.sleep(2)

        cv2.imshow("No S", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



    # When everything done, release the capture
    cv2.destroyAllWindows()