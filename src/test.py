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
import hashlib

import stack
from stack import Stack

if __name__ == '__main__':
    '''
    from paramiko import SSHClient
    from scp import SCPClient

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('192.168.0.100')

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    scp.put('test.txt',remote_path='/home/kristen/Desktop/')

    scp.close()
    '''

    temp_stack = Stack(4)

    for i in range(1, 6):
        temp_stack.push(i)


    while True:
        try:
            print(temp_stack.pop())
        except stack.Empty:
            break