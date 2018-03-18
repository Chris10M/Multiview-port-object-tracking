import numpy as np
import cv2
import threading
import time
import collections
import queue

from video_device import Frame
from tracker import Tracker, TrackFailure, TrackerPool
from events import Event
from utils import subtract_bounding_box
#from human_detect import BoundHuman
from boundhumantest import BoundHuman
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - ctime > 1:
            break
    cv2.destroyAllWindows()

    picked_list = BoundHuman.get_dict()
    print((picked_list))
    TrackerPool.push(picked_list)

    while True:
        frame = Frame.get()
        image = frame

        tracked_list = list()

        for tracker in TrackerPool.get():
            try:
                xa, ya, xb, yb = tracker.get_bounding_box(dtype=int)
                tracked_list.append((xa, ya, xb, yb))
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 0, 255 * tracker.get_id()), 2)

            except TrackFailure:
                print('{0} : {1}'.format(tracker.get_id(),tracker.is_alive()))
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255),
                            2)

        #for tracker in TrackerPool.get_dead():
        #    tracker.get_buffer()
        #    tracker.reclaim(picked_list[0])

        #picked_list = BoundHuman.get_list()

        #l = subtract_bounding_box(picked_list, tracked_list, threshold=10000)

        cv2.imshow("No S", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

