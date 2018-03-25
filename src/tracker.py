import cv2
import threading
import queue
import time

from events import Event
from video_device import VideoDevice, Frame

from uuid import getnode as get_mac
import time
import hashlib

def get_uuid(bounding_box):
    generated_id = get_mac() + time.time() + sum(bounding_box)
    uuid = hashlib.sha256(str(generated_id).encode('utf-8')).hexdigest()

    return uuid


class TrackFailure(Exception):
    pass


class Constant:
    TRACKING_FAILURE_DETECTED = -1
    TIME_OUT_SECOND_FOR_TRACK = 2
    QUEUE_BUFFER_SIZE = 50

    TRACKER_IMAGE_WIDTH = 60
    TRACKER_IMAGE_HEIGHT = 160




class Buffer:
    buffer_size = Constant.QUEUE_BUFFER_SIZE

    def __init__(self):
        self.buffer_queue = queue.Queue(Buffer.buffer_size)
        self.current_size = 0
        self.current_size_lock = threading.Lock()
    def put(self, item):
        try:
            self.buffer_queue.put_nowait(item)

            with self.current_size_lock:
                self.current_size += 1
        except queue.Full:
            self.buffer_queue.get_nowait()
            self.buffer_queue.task_done()

            self.buffer_queue.put_nowait(item)

    def get_all(self):
        items = list()

        with self.current_size_lock:
            self.current_size = 0

        while True:
            try:
                item = self.buffer_queue.get_nowait()
                items.append(item)

            except queue.Empty:
                return items

    def clear(self):

        with self.current_size_lock:
            if not self.current_size:
                return

        del self.buffer_queue
        self.buffer_queue = queue.Queue(Buffer.buffer_size)

class Tracker(threading.Thread):
    time_out_seconds = Constant.TIME_OUT_SECOND_FOR_TRACK

    POISON_PILL = (0, 0, 0, 0)

    def reclaim(self, bounding_box):
        _ = self.tracker.init(Frame.get(), tuple(bounding_box))
        self.track_failure.clear()

    def __init__(self, bounding_box):
        threading.Thread.__init__(self)
        self.tracker = cv2.TrackerMedianFlow_create()
        _ = self.tracker.init(Frame.get(), tuple(bounding_box))

        self.track_failure = threading.Event()
        self.track_failure.clear()
        self.id_lock = threading.Lock()

        self.buffer_queue = Buffer()
        self.bounding_box_queue = queue.Queue()

        with self.id_lock:
                self.id = get_uuid(bounding_box=bounding_box)

    def run(self):
        track_time = time.time()
        while True:
            frame = Frame.get()
            ok, bbox = self.tracker.update(frame)

            if ok:
                self.bounding_box_queue.put(bbox)

                tracker_frame = frame[int(bbox[1]):\
                                      int(bbox[3]),\
                                      int(bbox[0]):\
                                      int(bbox[2])]

                tracker_frame = cv2.resize(tracker_frame, (Constant.TRACKER_IMAGE_WIDTH, Constant.TRACKER_IMAGE_HEIGHT))
                tracker_frame = cv2.cvtColor(tracker_frame, cv2.COLOR_BGR2RGB)

                self.buffer_queue.put(tracker_frame)

                track_time = time.time()
                self.track_failure.clear()
            else:
                self.bounding_box_queue.put(Tracker.POISON_PILL)
                if time.time() - track_time > Tracker.time_out_seconds:
                    self.track_failure.set()
                    break

    def get_bounding_box(self, dtype=float):
        if self.track_failure.is_set():
            raise TrackFailure

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
    def push(bounding_box):
        thread = Tracker(bounding_box)
        thread.daemon = True

        thread.start()

        TrackerPool.tracker_list.append(thread)

    @staticmethod
    def get():
        return TrackerPool.tracker_list

    @staticmethod
    def get_dead():
        tracker_list = list()
        for tracker in TrackerPool.tracker_list:
            if not tracker.is_alive():
                tracker_list.append(tracker)

        return tracker_list

    @staticmethod
    def get_alive():
        tracker_list = list()
        for tracker in TrackerPool.tracker_list:
            if tracker.is_alive():
                tracker_list.append(tracker)

        return tracker_list

    @staticmethod
    def get_live_count():
        return len(TrackerPool.get_alive())

    @staticmethod
    def get_roi_for_alive():
        alive_tracker_list = TrackerPool.get_alive()

        alive_roi_list = list()

        for tracker in alive_tracker_list:
            xa, ya, xb, yb = tracker.get_bounding_box(dtype=int)

            alive_roi_list.append([tracker.get_id(), (xa, ya, xb, yb)])


        return alive_roi_list