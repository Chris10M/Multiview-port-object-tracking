import numpy as np
import cv2
import threading
import time
import collections
import queue

from video_device import Frame
#from tracker import Tracker, TrackFailure, TrackerPool, Buffer
from events import Event
from utils import subtract_bounding_box
#import human_detect
#import client


class Main:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        Event.bound_human_terminate.is_set()
        Frame.video.terminate()



class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1 / (time.time() - self.start)


class Threshold:
    new_object_frame_presence = 15
    track_failure_buffer_size = 1


with Main() as FrameLoop:
    print('Wait to initialize engine')
    print('initialization finished')

    import lightnet
    model = lightnet.load('tiny-yolo')
    while True:
        image = Frame.get()

        c = time.time()

        boxes = model(lightnet.Image(image.astype('float32')), thresh=0.5, hier_thresh=0.5, nms=0.45)

        print(c - time.time())
        #print(boxes)

        cv2.imshow("tracker", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




    cv2.destroyAllWindows()

    # picked_list = BoundHuman.get_dict()
    # print((picked_list))
    # TrackerPool.push(picked_list)

    '''

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

        for key, values in BoundHuman.get_dict().items():
            for box in values:
                xa ,ya, xb, yb = box
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 0, 255), 2)

        #for tracker in TrackerPool.get_dead():
        #    tracker.get_buffer()
        #    tracker.reclaim(picked_list[0])

        #picked_list = BoundHuman.get_list()

        #l = subtract_bounding_box(picked_list, tracked_list, threshold=10000)

        cv2.imshow("No S", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    '''