import cv2
import collections
import time
import numpy as np
from imutils.object_detection import non_max_suppression
import threading
from multiprocessing import Process, Queue, Pipe
import multiprocessing
from video_device import Frame
from events import Event


class Constant:
    SAMPLE_TIME_IN_SECONDS = 5
    REFRESH_TIME_IN_SECONDS = 2


class BoundHuman:
    bound_human_pipe_receive, bound_human_pipe_send = Pipe(duplex=False)
    frame_queue = multiprocessing.JoinableQueue()

    _get_list = None

    @staticmethod
    def get(frame_queue, bound_human_pipe_send):
        """
         groupThreshold – Minimum possible number of rectangles minus 1. The threshold is used in a group of rectangles to retain it.
         eps – Relative difference between sides of the rectangles to merge them into a group.

        """
        sample_time = Constant.SAMPLE_TIME_IN_SECONDS

        winStride = (4, 4)
        padding = (8, 8)
        scale = 1.05
        overlapThresh = 0.65

        groupThreshold = 10
        eps = 0.2

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


        def _sample_frame(frame_list, groupThreshold, eps):
            frame_list = np.array(frame_list)
            count = collections.defaultdict(int)

            for rects in frame_list:
                count[len(rects)] += 1

            count = [(k, count[k]) for k in sorted(count, key=count.get, reverse=True)]
            try:
                rect_length = count[0][0]
            except IndexError:
                return frame_list

            frame_rects_list = []

            for j in range(rect_length):
                rects = []
                try:
                    for i in range(len(frame_list)):
                        rects.append(frame_list[i][j].tolist())
                except IndexError:
                    continue
                rects, _ = cv2.groupRectangles(rects,
                                               groupThreshold=groupThreshold,
                                               eps=eps)

                for rect in rects:
                    frame_rects_list.append(rect)

            return frame_rects_list

        while True:
            start_time = time.time()
            pick_list = list()
            while True:
                image = frame_queue.get()

                (rects, weights) = hog. \
                                   detectMultiScale(image,
                                                    winStride=winStride,
                                                    padding=padding,
                                                    scale=scale)
                frame_queue.task_done()

                rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

                pick = non_max_suppression(rects,
                                           probs=None,
                                           overlapThresh=overlapThresh)

                if time.time() - start_time < sample_time:
                    pick_list.append(pick)
                else:
                    break

            picked_list =_sample_frame(pick_list,
                                       groupThreshold=groupThreshold,
                                       eps=eps)

            bound_human_pipe_send.send(picked_list)

    @staticmethod
    def update():
        get_list_lock = threading.Lock()
        while True:
            image = Frame.get()
            BoundHuman.frame_queue.put(image)

            try:
                get_list_lock.acquire()
                BoundHuman._get_list = BoundHuman.bound_human_pipe_receive.recv()
                get_list_lock.release()
            except:
                pass

    update_thread = threading.Thread(target=update.__func__)
    update_thread.daemon = True
    update_thread.start()

    get_process = Process(target=get.__func__, args=(frame_queue, bound_human_pipe_send))
    get_process.start()

    @staticmethod
    def get_list():
        while BoundHuman._get_list is None:
            continue

        return BoundHuman._get_list