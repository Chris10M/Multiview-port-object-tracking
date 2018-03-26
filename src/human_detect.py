import threading
from multiprocessing import Process, Queue, Pipe
import multiprocessing
from video_device import Frame
from events import Event
import colorsys
import os
import random
import cv2

import numpy as np
from collections import defaultdict
from imutils.object_detection import non_max_suppression


class BoundHuman:
    bound_human_queue = multiprocessing.JoinableQueue()
    frame_queue = multiprocessing.JoinableQueue()

    _get_list = None

    @staticmethod
    def get(frame_queue, bound_human_queue):

        class Path:
            model_data = os.path.join('.', 'person_detection', 'model_data', 'MobileNetSSD_deploy.caffemodel')
            proto_txt = os.path.join('.', 'person_detection', 'model_data', 'MobileNetSSD_deploy.prototxt.txt')


        class Parameter:
            score_threshold = 0.3
            iou_threshold = 0.5

        net = cv2.dnn.readNetFromCaffe(Path.proto_txt, Path.model_data)

        def get_bounding_box(frame):
            object_class_list = ["background", "aeroplane", "bicycle", "bird", "boat",
                                 "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                 "sofa", "train", "tvmonitor"]

            height, width, channels = frame.shape
            resized_image = cv2.resize(frame,  (300, 300), interpolation=cv2.INTER_LINEAR)

            blob = cv2.dnn.blobFromImage(resized_image, 0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            rect_list = list()
            score_list = list()
            predicted_class_list = list()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > Parameter.score_threshold:

                    idx = int(detections[0, 0, i, 1])

                    box = detections[0, 0, i, 3:7] * np.array([width, height,
                                                               width, height])

                    box = box.astype("int")
                    label = object_class_list[idx]

                    rect_list.append(tuple(box))
                    score_list.append(confidence)
                    predicted_class_list.append(label)

            predicted_class_dict = dict(zip(rect_list, predicted_class_list))

            rect_list = non_max_suppression(np.array(rect_list),
                                            probs=score_list,
                                            overlapThresh=0.3)

            nms_processed_predicted_class_dict = defaultdict(list)

            for rect, predicted_class in predicted_class_dict.items():
                if rect in rect_list:
                    nms_processed_predicted_class_dict[predicted_class].append(rect)

            return nms_processed_predicted_class_dict

        while True:
            image = frame_queue.get()

            class_bounding_box_dict = get_bounding_box(image)

            frame_queue.task_done()

            bound_human_queue.put(class_bounding_box_dict)


    @staticmethod
    def update():
        get_list_lock = threading.Lock()
        while True:
            image = Frame.get()
            BoundHuman.frame_queue.put(image)
            try:
                with get_list_lock:
                    BoundHuman._get_list = BoundHuman.bound_human_queue.get()

            except:
                pass
            if Event.bound_human_terminate.isSet():
                BoundHuman.get_process.terminate()
                break

    get_process = Process(target=get.__func__, args=(frame_queue, bound_human_queue))
    get_process.start()

    update_thread = threading.Thread(target=update.__func__)
    update_thread.daemon = True
    update_thread.start()



    @staticmethod
    def get_dict():
        while BoundHuman._get_list is None:
            continue

        return BoundHuman._get_list


class Classify:
    person = 'person'


def get_box_list():
    try:
        return  BoundHuman.get_dict()[Classify.person]
    except:
        return None