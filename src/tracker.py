import cv2
import threading
import queue
import time

from events import Event
from video_device import VideoDevice, Frame


class TrackFailure(Exception):
    pass


class Constant:
    TRACKING_FAILURE_DETECTED = -1
    TIME_OUT_SECOND_FOR_TRACK = 1
    QUEUE_BUFFER_SIZE = 50



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

    def reclaim(self, bounding_box):
        _ = self.tracker.init(Frame.get(), tuple(bounding_box))
        self.track_failure.clear()

    def __init__(self, bounding_box):
        threading.Thread.__init__(self)
        self.tracker = cv2.TrackerKCF_create()
        _ = self.tracker.init(Frame.get(), tuple(bounding_box))

        self.track_failure = threading.Event()
        self.track_failure.clear()
        self.id_lock = threading.Lock()

        self.buffer_queue = Buffer()
        self.bounding_box_queue = queue.Queue()

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
                self.track_failure.set()
                self.bounding_box_queue.put(Tracker.POISON_PILL)

            if self.track_failure.is_set():
                if time.time() - track_time > Tracker.time_out_seconds:
                    break
                elif ok:
                    self.track_failure.clear()

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
    def push(bounding_box_list):

        for roi in bounding_box_list:
            thread = Tracker(roi)
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

