import numpy as np
import cv2
import threading
import time
import collections
import queue

from video_device import Frame
from tracker import Tracker, TrackFailure, TrackerPool, Buffer
from events import Event
from utils import subtract_bounding_box
import human_detect
import client


class Main:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        Event.bound_human_terminate.is_set()
        Frame.video.terminate()

def pre_process_payload(image, detected_human_box_list, view_port_buffer, track_failure_frame_counter, new_object_frame_presence):
    new_roi = False
    track_failure  = False
    send_dead_tracker = False
    if track_failure_frame_counter > Threshold.track_failure_buffer_size:
        track_failure = True

    if new_object_frame_presence > Threshold.new_object_frame_presence:
        new_roi = True

    if track_failure_frame_counter or new_object_frame_presence:
        view_port_buffer.put((image, detected_human_box_list, TrackerPool.get_roi_for_alive()))

    if not track_failure_frame_counter and not new_object_frame_presence:
        view_port_buffer.clear()

    if track_failure or new_roi:
        send_payload = True
        if track_failure:
            send_dead_tracker = True
        else:
            send_dead_tracker = False
    else :
        send_payload = False

    if send_payload:
        if send_dead_tracker:
            #client.send_payload(view_port_buffer, TrackerPool.get_dead())
            pass
        else:
            client.send_payload(view_port_buffer, None)


class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)


class Threshold:
    new_object_frame_presence = 15
    track_failure_buffer_size = 1

with Main() as FrameLoop:

    print('Wait to initialize engine')
    time.sleep(5)
    print('initialization finished')

    #initial_track = human_detect.get_box_list()

    #for bounding_box in initial_track:
     #   TrackerPool.push(bounding_box)

    is_new_object_present = False
    new_object_frame_presence = 0

    view_port_buffer = Buffer()

    track_failure_frame_counter = 0

    while True:
        image = Frame.get()

        detected_human_box_list = human_detect.get_box_list()

        # A new person enters into the frame
        if TrackerPool.get_live_count() < len(detected_human_box_list):
            is_new_object_present = True
            #view_port_buffer.put((image, TrackerPool.get_roi_for_alive()))
        else:
            is_new_object_present = False
            new_object_frame_presence = 0
            #view_port_buffer.clear()

        if is_new_object_present is True:
            new_object_frame_presence += 1


            '''
            upload human_box_list and view_port_buffer
            create a new tracker
            '''
        #if person track is lost
        for tracker in TrackerPool.get():
            try:
                xa, ya, xb, yb = tracker.get_bounding_box(dtype=int)
                cv2.rectangle(image, (xa, ya), (xb, yb), (0, 0, 255), 2)
                track_failure_frame_counter = 0
            except TrackFailure:
                #view_port_buffer.put((image, TrackerPool.get_roi_for_alive()))
                track_failure_frame_counter += 1

        pre_process_payload(image=image,
                            detected_human_box_list=detected_human_box_list,
                            view_port_buffer=view_port_buffer,
                            track_failure_frame_counter=track_failure_frame_counter,
                            new_object_frame_presence=new_object_frame_presence)

            #print("lost track at {0}".format(tracker.id))

        cv2.imshow("tracker", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    #picked_list = BoundHuman.get_dict()
    #print((picked_list))
    #TrackerPool.push(picked_list)

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