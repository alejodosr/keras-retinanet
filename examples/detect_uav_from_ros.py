from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

import threading
from PIL import Image as ImagePIL

class image_converter:

    def __init__(self):
        self.FRAME_SKIP = 0
        self.SCALE_FACTOR = 1.0
        self.FPS = 30.0
        self.MAX_DELAY = 1.0 / self.FPS
        # use this environment flag to change which GPU to use
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        # set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(self.get_session())

        # load label to names mapping for visualization purposes
        self.labels_to_names = {0: 'drone', 1: 'bird', 2: 'car'}

        # adjust this to point to your downloaded/trained model
        # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
        self.model_path = os.path.join('/home/alejandro/py_workspace/keras-retinanet', 'snapshots', 'resnet50_csv_02_m100_inference.h5')

        # load retinanet model
        self.model = models.load_model(self.model_path, backbone_name='resnet50')

        self.graph = tf.get_default_graph()

        # if the model is not converted to an inference model, use the line below
        # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
        # self.model = models.convert_model(self.model)

        # print(self.model.summary())

        # Set ros subcriber
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image", Image, self.callback, queue_size=1)
        self.bb_pub = rospy.Publisher("bb", Int16MultiArray, queue_size=1)
        self.init = False
        self.high_freq_change = True

        # Other variables
        self.frame_count = 0

        # Preliminar loading
        image_temp = read_image_bgr('/home/alejandro/py_workspace/keras-retinanet/examples/2.jpg')

        # preprocess image for network
        image_temp = preprocess_image(image_temp)
        image_temp, scale = resize_image(image_temp)

        with self.graph.as_default():
            # process image
            start = time.time()
            boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image_temp, axis=0))
            print("processing time: ", time.time() - start)

            # correct for image scale
            boxes /= scale

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.8:
                    break

                print(label)

                b = box.astype(int)

        # Print information
        print("Successfully initiated...")


    def get_session(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        return tf.Session(config=self.config)

    def callback(self, data):
        # Increase frame count
        self.frame_count += 1

        if self.FRAME_SKIP != 0:
            self.FRAME_SKIP -= 1
            return
        else:
            self.FRAME_SKIP = 5

        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Debug
            # draw = self.image.copy()
        except CvBridgeError as e:
            print(e)

        if self.init:
            # Finding high frequency changes in image
            pass
        else:
            # It is firstly created in the callback
            self.prev_image = self.image.copy()

            # Found high frequency change
            self.high_freq_change = True

            # Restore variable
            self.init = True

        # It is firstly created in the callback
        self.prev_image = self.image.copy()

        if self.high_freq_change:
            # preprocess image for network
            self.image = preprocess_image(self.image)
            self.image, scale = resize_image(self.image)

            with self.graph.as_default():
                # process image
                start = time.time()
                boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(self.image, axis=0))
                print("processing time: ", time.time() - start)

                # self.FRAME_SKIP = int(self.FPS / ((time.time() - start) - self.MAX_DELAY))
                # print(self.FRAME_SKIP)

                # correct for image scale
                boxes /= scale

                # visualize detections
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    # scores are sorted so we can break
                    if score < 0.65:
                        break

                    print(label)

                    b = box.astype(int)
                    print(b)

                    if label == 0:
                            self.high_freq_change = False

                            bb = Int16MultiArray()
                            w_ext = int((b[2] - b[0]) * self.SCALE_FACTOR)
                            h_ext =  int((b[3] - b[1]) * self.SCALE_FACTOR)
                            bb.data = [b[0], b[1], w_ext, h_ext]
                            # w_ext = int(b[2] - b[0])
                            # h_ext =  int(b[3] - b[1])
                            # OFFSET = 20
                            # bb.data = [b[0] - OFFSET, b[1] - OFFSET, w_ext * 3, h_ext * 3]
                            self.bb_pub.publish(bb)

                            print(bb.data)
                            print(self.frame_count)

                            # Debug
                            # color = label_color(label)
                            # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                            # draw_box(draw, b, color=color)
                            # cv2.imshow("image", cv2.cvtColor(draw, cv2.COLOR_RGB2BGR))
                            # cv2.waitKey(0)



def main(args):
  rospy.init_node('image_detector', anonymous=True)

  ic = image_converter()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)