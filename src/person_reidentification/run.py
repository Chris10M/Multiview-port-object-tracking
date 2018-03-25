import tensorflow as tf
import numpy as np
import cv2
import os

'''
# image1 = cv2.imread('1.png')
image1 = cv2.resize(image1, (Pe_rid.IMAGE_WIDTH, Pe_rid.IMAGE_HEIGHT))
# image2 = cv2.imread('2.png')
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 160
image2 = cv2.resize(image2, (Pe_rid.IMAGE_WIDTH, Pe_rid.IMAGE_HEIGHT))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
'''
def process(viewport_buffer_queue, track_buffer, track_buffer_lock, roi_queue):

    class Pe_rid:
        batch_size = 1
        CHECKPOINT_DATA_PATH = os.path.join('.', 'model_data')

        IMAGE_WIDTH = 60
        IMAGE_HEIGHT = 160
        images = tf.placeholder(tf.float32, [2, batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
        is_train = tf.placeholder(tf.bool, name='is_train')
        sess = tf.Session()
        Inference = None

        @staticmethod
        def preprocess(images, is_train):
            def train():
                split = tf.split(images, [1, 1])
                shape = [1 for _ in range(split[0].get_shape()[1])]
                for i in range(len(split)):
                    split[i] = tf.reshape(split[i], [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])
                    split[i] = tf.image.resize_images(split[i], [Pe_rid.IMAGE_HEIGHT + 8, Pe_rid.IMAGE_WIDTH + 3])
                    split[i] = tf.split(split[i], shape)
                    for j in range(len(split[i])):
                        split[i][j] = tf.reshape(split[i][j], [Pe_rid.IMAGE_HEIGHT + 8, Pe_rid.IMAGE_WIDTH + 3, 3])
                        split[i][j] = tf.random_crop(split[i][j], [Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])
                        split[i][j] = tf.image.random_flip_left_right(split[i][j])
                        split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                        split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                        split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                        split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                        split[i][j] = tf.image.per_image_standardization(split[i][j])
                return [tf.reshape(tf.concat(split[0], axis=0), [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3]),
                    tf.reshape(tf.concat(split[1], axis=0), [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])]
            def val():
                split = tf.split(images, [1, 1])
                shape = [1 for _ in range(split[0].get_shape()[1])]
                for i in range(len(split)):
                    split[i] = tf.reshape(split[i], [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])
                    split[i] = tf.image.resize_images(split[i], [Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH])
                    split[i] = tf.split(split[i], shape)
                    for j in range(len(split[i])):
                        split[i][j] = tf.reshape(split[i][j], [Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])
                        split[i][j] = tf.image.per_image_standardization(split[i][j])
                return [tf.reshape(tf.concat(split[0], axis=0), [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3]),
                    tf.reshape(tf.concat(split[1], axis=0), [Pe_rid.batch_size, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3])]
            return tf.cond(is_train, train, val)

        @staticmethod
        def network(images1, images2, weight_decay):
            with tf.variable_scope('network'):
                # Tied Convolution
                conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
                pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
                conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
                pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
                conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
                pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
                conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
                pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

                # Cross-Input Neighborhood Differences
                trans = tf.transpose(pool1_2, [0, 3, 1, 2])
                shape = trans.get_shape().as_list()
                m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
                reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
                f = tf.multiply(reshape, m1s)

                trans = tf.transpose(pool2_2, [0, 3, 1, 2])
                reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
                g = []
                pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
                for i in range(shape[2]):
                    for j in range(shape[3]):
                        g.append(pad[:,:,:,i:i+5,j:j+5])

                concat = tf.concat(g, axis=0)
                reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
                g = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
                reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
                reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
                k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
                k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

                # Patch Summary Features
                l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
                l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

                # Across-Patch Features
                m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
                pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
                m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
                pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

                # Higher-Order Relationships
                concat = tf.concat([pool_m1, pool_m2], axis=3)
                reshape = tf.reshape(concat, [Pe_rid.batch_size, -1])
                fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
                fc2 = tf.layers.dense(fc1, 2, name='fc2')

                return fc2

        @staticmethod
        def init():
            weight_decay = 0.0005

            images1, images2 = Pe_rid.preprocess(Pe_rid.images, Pe_rid.is_train)

            print('Build network')
            logits =  Pe_rid.network(images1, images2, weight_decay)
            Pe_rid.Inference = tf.nn.softmax(logits)

            Pe_rid.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(Pe_rid.CHECKPOINT_DATA_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restore model')
                saver.restore(Pe_rid.sess, ckpt.model_checkpoint_path)

        def detect(self, image_1, image_2):

            test_images = np.array([image_1, image_2])

            feed_dict = {Pe_rid.images: test_images, Pe_rid.is_train: False}
            prediction = Pe_rid.sess.run(Pe_rid.Inference, feed_dict=feed_dict)
            print(prediction)
            print(bool(not np.argmax(prediction[0])))

        def __enter__(self):
            Pe_rid.init()

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            Pe_rid.sess.close()


    def is_same_roi(alive_roi_box, detected_roi):
        return alive_roi_box == detected_roi
        pass

    with Pe_rid() as perid_instance:
        while True:
            viewport_buffer = viewport_buffer_queue.get()
            roi_dict = dict()

            for image, detected_human_box_list, alive_roi_tracker_list in viewport_buffer:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                not_tracked_roi = list()

                alive_track_id_list = list()

                for detected_roi in detected_human_box_list:
                    for track_id, alive_roi_box in alive_roi_tracker_list:
                        alive_track_id_list.append(track_id)
                        if is_same_roi(alive_roi_box, detected_roi):
                            break
                    else:
                        not_tracked_roi.append(detected_roi)


                for roi in not_tracked_roi:
                    xa, ya, xb, yb = roi

                    roi_image = image[ya:ya + (yb - ya), xa:xa + (xb - xa)]
                    roi_image = cv2.resize(roi_image, (Pe_rid.IMAGE_WIDTH, Pe_rid.IMAGE_HEIGHT))

                    with track_buffer_lock:
                        for track_id, track_frame in track_buffer.items():
                            if track_id not in alive_track_id_list:
                                hit_threshold = 5
                                hit_counter = 0
                                for frame in track_frame:
                                    frame = np.reshape(frame, (1, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3)).astype(float)
                                    roi_image = np.reshape(roi_image, (1, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3)).astype(float)
                                    if perid_instance.detect(image, frame):
                                        hit_counter += 1

                                if hit_counter > hit_threshold:
                                    roi_dict[roi] = track_id
                                    break
                        else:
                            roi_dict[roi] = None

            roi_queue.put(roi_dict)



import time

class Benchmark:
    def __init__(self):
        self.start = time.time()

    def end(self):
        return 1/(time.time() - self.start)

if __name__ == '__main__':
    import threading

    image1 = cv2.imread('1.png')
    image1 = cv2.resize(image1, (Pe_rid.IMAGE_WIDTH, Pe_rid.IMAGE_HEIGHT))
    image2 = cv2.imread('2.png')
    image2 = cv2.resize(image2, (Pe_rid.IMAGE_WIDTH, Pe_rid.IMAGE_HEIGHT))

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = np.reshape(image1, (1, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3)).astype(float)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = np.reshape(image2, (1, Pe_rid.IMAGE_HEIGHT, Pe_rid.IMAGE_WIDTH, 3)).astype(float)

    with Pe_rid() as t:
        l = Benchmark()
        thread_list = list()
        for _ in range(100):
            thread = threading.Thread(target=t.detect, args=(image1, image2))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()
            #t.detect(image1,image2)

        print(l.end())