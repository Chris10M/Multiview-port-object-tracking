import pickle
import os

from stack import Stack


class ServerFileSystem:
    payload_path = os.path.join('.', 'payload')
    client_list = os.path.join('.', 'client_list.json')

class Payload:
    def __init__(self, payload_file):
        payload = pickle.load(payload_file)

        self.view_port_buffer = payload.view_port_buffer
        self.is_tracker_present = payload.is_tracker_present
        self.track_buffer = payload.track_buffer

    def get_view_port_buffer(self):
        return self.view_port_buffer

    def get_track_buffer(self):
        return self.track_buffer




import os
import threading
import collections

class FileUpdateNotifier:
    file_handle_dict = collections.defaultdict(list)
    file_update_request_dict = dict()
    file_update_request_lock = threading.Lock()
    snapshot_handler_dict = collections.defaultdict(int)
    new_file_generator_lock = threading.Lock()
    terminate_event = threading.Event()
    file_handle_dict_lock = threading.Lock()
    number_of_files = 0
    new_file_event = threading.Event()
    new_file_acknowledged_event = threading.Event()
    file_handle_length_dict = collections.defaultdict(int)

    run_thread = None

    @staticmethod
    def is_new_file_present():
        while True:
            if FileUpdateNotifier.terminate_event.is_set():
                break

            file_list = list()
            for (_, _, file_name) in os.walk(os.path.join(ServerFileSystem.payload_path)):
                file_list.extend(file_name)
                break

            number_of_files = len(file_list)

            if FileUpdateNotifier.number_of_files != number_of_files:
                FileUpdateNotifier.new_file_event.set()
                #print("Before_lock cur:{0}, old:{1}".format(number_of_files, FileUpdateNotifier.number_of_files))

                FileUpdateNotifier.new_file_acknowledged_event.wait(timeout=.5)
                FileUpdateNotifier.new_file_acknowledged_event.clear()

                FileUpdateNotifier.number_of_files = number_of_files

                #print("After_lock cur:{0}, old:{1}".format(number_of_files, FileUpdateNotifier.number_of_files))



    @staticmethod
    def __get_update():
        file_list = list()

        for (_, _, file_name) in os.walk(os.path.join(ServerFileSystem.payload_path)):
            file_list.extend(file_name)
            break
        for file_name in file_list:
            file_name = file_name.strip('.pickle').split('_')
            #print("file_name = file_name.strip('.pickle').split('_'")
            #print(file_name)

            if file_name[1] not in FileUpdateNotifier.file_handle_dict[file_name[0]]:
                FileUpdateNotifier.file_handle_dict[file_name[0]].append(file_name[1])
                FileUpdateNotifier.file_update_request_dict[file_name[0]] = True
                #print('FileUpdateNotifier.file_update_request_dict[file_name[0]] = True')


        new_file_dict = collections.defaultdict(list)
        for handle, is_update in FileUpdateNotifier.file_update_request_dict.items():
            if is_update is True:
                old_len = FileUpdateNotifier.file_handle_length_dict[handle]
                cur_len = len(FileUpdateNotifier.file_handle_dict[handle])
                #print(FileUpdateNotifier.file_handle_dict[handle])
                new_file_dict[handle].extend(FileUpdateNotifier.file_handle_dict[handle][old_len: cur_len])

                FileUpdateNotifier.file_handle_length_dict[handle] = cur_len
                FileUpdateNotifier.file_update_request_dict[handle] = False

        return new_file_dict

    @staticmethod
    def get_new_files():
        new_file_dict = dict()

        if FileUpdateNotifier.new_file_event.is_set():
            FileUpdateNotifier.new_file_acknowledged_event.set()

            new_file_dict = FileUpdateNotifier.__get_update()

            FileUpdateNotifier.new_file_event.clear()

        return new_file_dict

    @staticmethod
    def terminate():
        FileUpdateNotifier.terminate_event.set()


    @staticmethod
    def start():
        FileUpdateNotifier.run_thread = threading.Thread(target=FileUpdateNotifier.is_new_file_present)
        FileUpdateNotifier.run_thread.start()


def get_new_files():
        hostname_filename_dict = collections.defaultdict(list)

        for hostname, timestamp_list in FileUpdateNotifier.get_new_files().items():

            for timestamp in timestamp_list:
                hostname_filename_dict[hostname].append('{0}_{1}.pickle'.format(hostname, timestamp))

        return hostname_filename_dict

import multiprocessing

shared_memory = multiprocessing.Manager()

class TrackFrameDatabase:
    '''
    dict(hash_id) = [Frames]
    '''
    #track_frame_dict = collections.defaultdict(list)
    track_frame_dict = shared_memory.dict()
    track_frame_dict_lock = shared_memory.Lock()

    view_port_frame_dict = shared_memory.dict()
    view_port_frame_dict_lock = threading.Lock()

    recently_used_host = Stack()
    recently_used_track_id = Stack()
    recently_used_lock = threading.Lock()

    new_file_count_lock = threading.Lock()
    new_file_count = 0

    __update_track_frame_thread = None
    __update_track_frame_terminate_event = threading.Event()

    file_updated_event = threading.Event()

    @staticmethod
    def __update_track_frame():

        while True:
            if TrackFrameDatabase.__update_track_frame_terminate_event.is_set():
                break

            hostname_filename_dict = get_new_files()

            if not hostname_filename_dict:
                continue

            with TrackFrameDatabase.new_file_count_lock:
                TrackFrameDatabase.new_file_count += len(hostname_filename_dict.values())

            for hostname, filename_list in hostname_filename_dict.items():
                for file_name in filename_list:

                    with open(os.path.join(ServerFileSystem.payload_path, file_name), 'rb') as pickle_file:
                        payload = Payload(pickle_file)

                    with TrackFrameDatabase.track_frame_dict_lock:
                        for track_id, tracker_buffer in payload.get_track_buffer():
                            try:
                                TrackFrameDatabase.track_frame_dict[payload.track_id].extend(tracker_buffer)
                            except:
                                TrackFrameDatabase.track_frame_dict[payload.track_id] = tracker_buffer

                            with TrackFrameDatabase.recently_used_lock:
                                TrackFrameDatabase.recently_used_track_id.push(track_id)

                    with TrackFrameDatabase.view_port_frame_dict_lock:
                        TrackFrameDatabase.view_port_frame_dict[hostname] = payload.get_view_port_buffer()

                with TrackFrameDatabase.recently_used_lock:
                    TrackFrameDatabase.recently_used_host.push(hostname)

            TrackFrameDatabase.file_updated_event.set()

    @staticmethod
    def get_track_id_iterator():
        with TrackFrameDatabase.recently_used_lock:
            for track_id in TrackFrameDatabase.recently_used_track_id.yield_generator():
                with TrackFrameDatabase.track_frame_dict_lock:
                    yield TrackFrameDatabase.track_frame_dict[track_id]

    @staticmethod
    def get_latest_host_update():

        TrackFrameDatabase.file_updated_event.wait()

        while TrackFrameDatabase.new_file_count > 0:

            with TrackFrameDatabase.recently_used_lock:
                with TrackFrameDatabase.view_port_frame_dict_lock:
                    with TrackFrameDatabase.new_file_count_lock:
                        TrackFrameDatabase.new_file_count -= 1

                    hostname = TrackFrameDatabase.recently_used_host.pop()
                    return hostname, TrackFrameDatabase.view_port_frame_dict[hostname]



        TrackFrameDatabase.file_updated_event.clear()

    @staticmethod
    def start():
        TrackFrameDatabase.__update_track_frame_terminate_event.clear()

        TrackFrameDatabase.__update_track_frame_thread = threading.Thread(target=TrackFrameDatabase.__update_track_frame)
        TrackFrameDatabase.__update_track_frame_thread.start()

    @staticmethod
    def terminate():
        TrackFrameDatabase.__update_track_frame_terminate_event.set()





#def process(viewport_buffer, track_buffer, data_sent_event):

#import time
FileUpdateNotifier.start()
TrackFrameDatabase.start()

import person_reidentification.run as perid
import json

class Client:
    client_list = list()

    def __init__(self, client):
        self.hostname = client['hostname']
        self.address = client['address']

        Client.client_list.append(self)

    def get(self):
        return self.hostname, self.address

    @staticmethod
    def get_list():
        return Client.client_list

def populate_client():
    with open(ServerFileSystem.client_list, 'r') as client_list_file:
        client_list_json = json.load(client_list_file)

        for client in client_list_json:
            Client(client)




#l = manager.list(range(10))

class Perid:
    def __init__(self):
        self.detected_roi = multiprocessing.Queue()
        self.viewport_buffer = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=perid.process, args=(self.viewport_buffer,
                                                                           TrackFrameDatabase.track_frame_dict,
                                                                           TrackFrameDatabase.track_frame_dict_lock,
                                                                           self.detected_roi))
        self.process.start()

    def detect(self, viewport_buffer):
        self.viewport_buffer.put(viewport_buffer)
        return self.detected_roi.get()


class PeridProcessPool:
    populate_client()
    client_list = dict([ x.get() for x in Client.get_list()])
    process_dict = dict()

    @staticmethod
    def start():
        for hostname, address in PeridProcessPool.client_list.items():
            PeridProcessPool.process_dict[hostname] = Perid()

    @staticmethod
    def detect():
        host_roi_dict = dict()
        hostname, host_viewport_buffer = TrackFrameDatabase.get_latest_host_update()
        print(hostname)
        host_roi_dict[hostname] = PeridProcessPool.process_dict[hostname].detect(host_viewport_buffer)

        return host_roi_dict

PeridProcessPool.start()
import pickletools
from paramiko import SSHClient
from scp import SCPClient
import io
import time

def send_response(hostname, address, roi):

    client_path = os.path.join('/', 'home', 'pi', 'Multiview-port-object-tracking', 'src', 'response')

    payload_pickle = pickletools.optimize(pickle.dumps(roi))

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(address, username='pi', password='raspberry')

    scp = SCPClient(ssh.get_transport())

    # generate in-memory file-like object
    fl = io.BytesIO()
    fl.write(payload_pickle)
    fl.seek(0)
    # upload it directly from memory
    scp.putfo(fl, os.path.join(client_path, 'response_{0}.pickle'.format(time.time())))

    # close connection
    scp.close()
    # close file handler
    fl.close()

while True:

    host_roi_dict = PeridProcessPool.detect()

    try:
        print(host_roi_dict)
    for hostname in host_roi_dict.keys():
        send_response(hostname, PeridProcessPool.client_list[hostname], host_roi_dict[hostname])
    except:
        pass

    #p.join()

#for i in TrackFrameDatabase.get_latest_host_update():

    #print(i)
    #break

FileUpdateNotifier.terminate()
TrackFrameDatabase.terminate()

'''
import cv2

while True:
    hostname, view_port_buffer = TrackFrameDatabase.get_latest_host_update()

    for frame in view_port_buffer:
        image, frame_roi_list, alive_tracker_roi = frame

        for roi in frame_roi_list:
            xa, ya, xb, yb = roi
            image_roi = image[ya:yb, xa:xb]




        cv2.waitKey(0)

    cv2.destroyAllWindows()


    print(detected_human_box_list)
    print(alive_tracker_roi)

FileUpdateNotifier.terminate()


exit()

with open('test.pickle', 'rb') as s:
    import zlib
    import gzip
    import pickletools
    t = pickletools.optimize(pickle.dumps(Payload(s)))
    t = zlib.compress(t, 9)
    with gzip.open('file.pickle.gz', 'wb') as f:
        f.write(t)
    with gzip.open('file.pickle.gz', 'rb') as f:
        t = f.read()
        t = zlib.decompress(t)
        t = pickle.loads(t)
    import cv2

    print(t.track_id)

    for image in t.get_track_buffer():
        cv2.imshow('detected', image)
        cv2.waitKey(0)


    cv2.destroyAllWindows()
'''