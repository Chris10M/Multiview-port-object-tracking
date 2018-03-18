import colorsys
import os
import random
import cv2

import numpy as np
from keras import backend as K
from keras.models import load_model
from collections import defaultdict
from imutils.object_detection import non_max_suppression

from person_detection.yad2k.models.keras_yolo import yolo_eval, yolo_head


class Path:
    model_path = os.path.join('.', 'person_detection', 'model_data', 'tiny.h5')
    anchors_path = os.path.join('.', 'person_detection', 'model_data', 'tiny_anchors.txt')
    classes_path = os.path.join('.', 'person_detection', 'model_data', 'pascal_classes.txt')


class Parameter:
    score_threshold = 0.3
    iou_threshold = 0.5

    with open(Path.classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(Path.anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)


class Init:
    sess = K.get_session()

    yolo_model = load_model(Path.model_path)

    # Verify model, anchors, and classes are compatible
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == len(Parameter.anchors) * (len(Parameter.class_names) + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(Parameter.class_names), 1., 1.)
                  for x in range(len(Parameter.class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, Parameter.anchors, len(Parameter.class_names))
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=Parameter.score_threshold,
        iou_threshold=Parameter.iou_threshold)


def get_bounding_box(frame):
    height, width, channels = frame.shape
    resized_image = cv2.resize(frame, Init.model_image_size,interpolation = cv2.INTER_LINEAR)

    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    out_boxes, out_scores, out_classes = Init.sess.run(
        [Init.boxes, Init.scores, Init.classes],
        feed_dict={
            Init.yolo_model.input: image_data,
            Init.input_image_shape: [height, width],
            K.learning_phase(): 0
        })
    rect_list = list()
    score_list = list()
    predicted_class_list = list()
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = Parameter.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
        right = min(width, np.floor(right + 0.5).astype('int32'))
        rect_list.append((left, top, right, bottom))
        score_list.append(score)
        predicted_class_list.append(predicted_class)

    predicted_class_dict = dict(zip(rect_list, predicted_class_list))

    rect_list = non_max_suppression(np.array(rect_list),
                                probs=score_list,
                                overlapThresh=0.3)

    nms_processed_predicted_class_dict = defaultdict(list)

    for rect, predicted_class in predicted_class_dict.items():
        if rect in rect_list:
            nms_processed_predicted_class_dict[predicted_class].append(rect)

    return nms_processed_predicted_class_dict


class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)



def _main():

    file_name = 'test1.mp4'
    cap = cv2.VideoCapture(file_name)
    while cap.isOpened():
        _, frame = cap.read()

        print(get_bounding_box(frame))

        cv2.imshow('detected', frame)
        cv2.waitKey(1)

    Init.sess.close()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    _main()
