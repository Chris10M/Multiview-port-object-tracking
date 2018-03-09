import cv2
import threading
from main import Frame, Events
import queue
import time


class Constants:
    TRACKING_FAILURE_DETECTED = -1
    TIME_OUT_SECOND_FOR_TRACK = 1


class Tracker(threading.Thread):
    time_out_seconds = Constants.TIME_OUT_SECOND_FOR_TRACK

    def __init__(self, bounding_box):
        threading.Thread.__init__(self)
        self.tracker = cv2.TrackerMedianFlow_create()
        self.bounding_box_queue = queue.Queue()
        ok = self.tracker.init(Frame.get(), bounding_box)


    def run(self):
        track_time = time.time()
        while True:
            frame = Frame.get()
            ok, bbox = self.tracker.update(frame)

            if ok:
                self.bounding_box_queue.put(bbox)
                Events.track_failure.clear()
                track_time = time.time()
            else:
                Events.track_failure.set()

            if Events.track_failure.is_set():
                if time.time() - track_time > Tracker.time_out_seconds:
                    break

    def get_bounding_box(self):
        bounding_box = self.bounding_box_queue.get()
        self.bounding_box_queue.task_done()

        return bounding_box

    def get_bounding_box_as_rect(self):
        bounding_box = self.bounding_box_queue.get()
        self.bounding_box_queue.task_done()

        point_1 = (int(bounding_box[0]), int(bounding_box[1]))

        point_2 = (int(bounding_box[0] + bounding_box[2]),
                   int(bounding_box[1] + bounding_box[3]))

        return point_1, point_2




if __name__ == '__main__':
    tracker = cv2.TrackerGOTURN_create()

    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break


        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
