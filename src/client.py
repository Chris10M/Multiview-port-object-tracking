import numpy as np
import io
import pickletools
import pickle
from paramiko import SSHClient
from scp import SCPClient
import os
from uuid import getnode as get_mac
import datetime

class Server:
    address = '10.0.0.7'
    pickle_directory = os.path.join('/', 'home', 'kristen', 'Desktop', 'server_pickle')
    pickle_name = '{0}_{1}.pickle'.format(hex(get_mac()), datetime.datetime.now())


class Payload:

    def __init__(self, view_port_buffer, tracker_list):
        self.view_port_buffer = view_port_buffer.get_all()
        self.is_tracker_present = True if tracker_list is not None else False
        self.track_buffer = list()

        if self.is_tracker_present:
            for tracker in tracker_list:
                tracker_buffer = tracker.buffer_queue.get_all()

                if tracker_buffer:
                    self.track_buffer.append((tracker.get_id(), tracker_buffer))

    def serialize(self):
        return pickletools.optimize(pickle.dumps(self))


def send_payload(view_port_buffer, tracker):
    paylod_pickle = Payload(view_port_buffer, tracker).serialize()
    
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(Server.address)

    scp = SCPClient(ssh.get_transport())

    # generate in-memory file-like object
    fl = io.BytesIO()
    fl.write(paylod_pickle)
    fl.seek(0)
    # upload it directly from memory
    scp.putfo(fl, os.path.join(Server.pickle_directory, Server.pickle_name))

    # close connection
    scp.close()
    # close file handler
    fl.close()
    '''
    t = pickle.loads(paylod_pickle)
    import cv2
    for image, roi_list, tracker_live_roi in (t.view_port_buffer):
        cv2.imshow('detected', image)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
    '''
    exit()

'''
def main:
    view_port_buffer = list()
    tracker = list()
    i = 0
    file_name = 'test1.mp4'
    import cv2
    import time
    import os
    file_path = os.path.join('.', 'outputs')
    files = list()
    for (_, _, file) in os.walk(file_path):
        files.extend(file)
        break
    files = sorted(files, key=lambda x:int(x.strip('.jpg')))

    for file in files:
        frame = cv2.imread(os.path.join(file_path, file))
        mask = np.zeros((416, 416, 3), dtype="uint8")

        height, width, _ = frame.shape
       # cv2.rectangle(mask,(0, 0), (width,height),(255,255,255), -1)
        #cv2.imshow('new', mask)

        #frame = cv2.bitwise_and(mask,mask,mask=frame)
        #cv2.imshow('detected', frame)

        mask[ :height, :width] = frame

        view_port_buffer.append(mask)
        tracker.append(mask)


    s = pickle.dumps(view_port_buffer)
    s = pickletools.optimize(s)

    import io
    import pickletools
    import pickle
    from paramiko import SSHClient
    from scp import SCPClient
    import scp as sc

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('192.168.0.100')

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    # generate in-memory file-like object
    fl = io.BytesIO()
    fl.write(s)
    fl.seek(0)
    # upload it directly from memory
    scp.putfo(fl, '/home/kristen/Desktop/test.pickle')

    # close connection
    scp.close()
    # close file handler
    fl.close()


    
    def make_img_list_same_shape(self, image_list):
        new_image_list = list()

        for img in image_list:
            mask = np.zeros(Payload.tracker_payload_shape, dtype="uint8")
            height, width, _ = img.shape
            mask[ :height, :width] = img

            new_image_list.append(mask)

        return new_image_list
    
    '''


import socket
print(Server.pickle_name)

