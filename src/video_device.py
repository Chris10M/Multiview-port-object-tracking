import cv2
import multiprocessing as mp
import queue

from events import Event

class Constant:
    CAMERA = 0
    BUFFER_SIZE = 16


class VideoDevice:
    buffer_size = Constant.BUFFER_SIZE

    def __init__(self, device=Constant.CAMERA):
        self.frame_queue = mp.JoinableQueue(self.buffer_size)
        self.device = device
        self.frame_process = mp.Process(target=self._run, args=(self.device, ))

    def _run(self, device):
        cap = cv2.VideoCapture(device)
        while cap.isOpened():
            _, frame = cap.read()
            self.frame_queue.put(obj=frame, block=True, timeout=None)

    def start(self):
        self.frame_process.start()

    def get_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return None

        self.frame_queue.task_done()

        return frame

    def terminate(self):
        self.frame_process.terminate()


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


if __name__ == '__main__':
    file_name = 'test2.mp4'
    video = VideoDevice(device=file_name)
    video.start()

    while (True):

        frame = video.get_frame()

        if frame is None:
            continue

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()



