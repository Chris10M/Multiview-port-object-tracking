import pickle
import pickletools
from stack import Stack

class Payload:

    def __init__(self, payload_file):
        payload = pickle.load(payload_file)

        self.view_port_buffer = payload.view_port_buffer
        self.is_tracker_present = payload.is_tracker_present
        self.track_buffer = payload.track_buffer
        self.track_id = payload.track_id

    def get_view_port_buffer(self):
        return self.view_port_buffer

    def get_track_buffer(self):
        return self.track_buffer

    def get_track_id(self):
        return self.track_id


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
            for (_, _, file_name) in os.walk(os.path.join('.', 'stest')):
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

        for (_, _, file_name) in os.walk(os.path.join('.', 'stest')):
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


class TrackFrameDatabase:
    '''
    dict(hash_id) = [Frames]
    '''

    track_frame_dict = collections.defaultdictdict(list)
    track_frame_dict_lock = threading.Lock()

    view_port_frame_dict = dict()
    view_port_frame_dict_lock = threading.Lock()

    recently_used_host = Stack()
    recently_used_track_id = Stack()
    recently_used_lock = threading.Lock()

    new_file_count_lock = threading.Lock()
    new_file_count = 0

    __update_track_frame_thread = None
    __update_track_frame_terminate_event = threading.Event()

    @staticmethod
    def __update_track_frame():

        while True:
            if TrackFrameDatabase.__update_track_frame_terminate_event.is_set():
                break

            hostname_filename_dict = get_new_files()

            with TrackFrameDatabase.new_file_count_lock:
                TrackFrameDatabase.new_file_count += len(hostname_filename_dict.values())

            for hostname, filename_list in hostname_filename_dict.items():
                for file_name in filename_list:
                    payload = get_payload_from_file(file_name)

                    with TrackFrameDatabase.track_frame_dict_lock:
                        TrackFrameDatabase.track_frame_dict[payload.get_track_id()].append(payload.get_track_buffer())

                    with TrackFrameDatabase.view_port_frame_dict_lock:
                        TrackFrameDatabase.view_port_frame_dict[hostname] = payload.get_view_port_buffer()

                    with TrackFrameDatabase.recently_used_lock:
                        TrackFrameDatabase.recently_used_track_id.push(payload.get_track_id)

                with TrackFrameDatabase.recently_used_lock:
                    TrackFrameDatabase.recently_used_host.push(hostname)



    def track_id_iterator(self):
        with TrackFrameDatabase.recently_used_lock:
            for track_id in TrackFrameDatabase.recently_used_track_id.yield_generator():
                with TrackFrameDatabase.track_frame_dict_lock:
                    yield TrackFrameDatabase.track_frame_dict[track_id]

    def get_latest_host_update(self):

        while TrackFrameDatabase.new_file_count > 0:

            with TrackFrameDatabase.recently_used_lock:
                with TrackFrameDatabase.view_port_frame_dict_lock:
                    yield TrackFrameDatabase.view_port_frame_dict[TrackFrameDatabase.recently_used_host.pop()]

            with TrackFrameDatabase.new_file_count_lock:
                TrackFrameDatabase.new_file_count -= 1

    @staticmethod
    def start():
        TrackFrameDatabase.__update_track_frame_terminate_event.clear()

        TrackFrameDatabase.__update_track_frame_thread = threading.Thread(target=TrackFrameDatabase.__update_track_frame)
        TrackFrameDatabase.__update_track_frame_thread.start()

    @staticmethod
    def terminate():
        TrackFrameDatabase.__update_track_frame_terminate_event.set()



class PeridProcessPool:
    size = 5

    @staticmethod
    def start():






exit()
import time
FileUpdateNotifier.start()
while True:
    pass
    def get_new_files():
        file_list = list()
        for hostname, timestamp_list in FileUpdateNotifier.get_new_files().items():
            for timestamp in timestamp_list:
                file_list.append('{0}_{1}.pickle'.format(hostname, timestamp))

        return file_list

    l = get_new_files()
    if l:
        print(l)

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