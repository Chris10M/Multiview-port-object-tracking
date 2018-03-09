from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import threading

import time
import collections
import queue

if __name__ == '__main__':

