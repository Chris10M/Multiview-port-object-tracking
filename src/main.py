import numpy as np
import cv2
import threading
import time
import collections
import queue

from video_device import Frame
from human_detect import BoundHuman
from tracker import Tracker, TrackFailure
from events import Event

class Main:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        Frame.video.terminate()


class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)


with Main() as FrameLoop:
    ctime = time.time()

    while True:
        image = Frame.get()
        cv2.imshow("No EDIT", image)
        if cv2.waitKecy(1) & 0xFF == ord('q'):
            break

        if time.time() - ctime > 1:
            break
    cv2.destroyAllWindows()


    picked_list = BoundHuman.get()


    class TrackerPool:
        tracker_list = list()

        @staticmethod
        def get_by_id(_id):
            for tracker in TrackerPool.tracker_list:
                if tracker.get_id() == _id:
                    return tracker
            else:
                return None


        @staticmethod
        def push(bounding_box_list):

            for roi in bounding_box_list:
                thread = Tracker(roi)
                thread.daemon = True

                thread.start()

                TrackerPool.tracker_list.append(thread)

        @staticmethod
        def get():
            return TrackerPool.tracker_list




        def __enter__(self):
            for roi in self.picked_list:
                thread = Tracker(roi)
                thread.daemon = True
                thread.start()
                self.roi_list.append(thread)



    roi_list = list()

    for roi in picked_list:
        thread = Tracker(roi)
        thread.daemon = True
        thread.start()
        roi_list.append(thread)

    ##       print(roi_box)

    while True:
        frame = Frame.get()
        image = frame
        k = False
        for tracked_roi in roi_list:
            b = Benchmark()
            try:
                xa, ya, xb, yb = tracked_roi.get_bounding_box(dtype=int)
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 0, 255 * tracked_roi.get_id()), 2)

            except TrackFailure:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255),
                            2)
                k=True
        if k:
            break
            0, 1, 2, 3
            print(b.end())
            print(tracked_roi.get_id())
        # time.sleep(2)

        cv2.imshow("No S", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # When everything done, release the capture
    cv2.destroyAllWindows()

