import numpy as np
import sys


def subtract_bounding_box(bounding_box_list_1, bounding_box_list_2, threshold):
    """

    :param bounding_box_list_1:
    :param bounding_box_list_2:
    :param threshold:
    :return: bounding_box_list_1 - bounding_box_list_2
    """
    max_value = sys.maxsize
    value_1 = list()

    for rect_1 in bounding_box_list_1:
        min_value = max_value
        for rect_2 in bounding_box_list_2:
            rect_1 = np.array(rect_1)
            rect_2 = np.array(rect_2)

            dist = np.linalg.norm(rect_1 - rect_2)
            if dist < min_value:
                min_value = dist
        value_1.append(min_value)

    value_1 = dict(enumerate(value_1))

    print(value_1)

    unique_bounding_box = list()
    for key, value in value_1.items():
        if value < threshold:
            unique_bounding_box.append(bounding_box_list_1[key])

    return unique_bounding_box


