import threading

class Event:
    frame_ready = threading.Event()
    ready_to_track = threading.Event()
    track_failure = threading.Event()