# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# load label to names mapping for visualization purposes
# labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
SHOW_IMAGES = 1
SAVE_ANNOTATIONS = 0
# LABELS_TO_NAMES = {0: 'drone', 1: 'bird', 2: 'car'}
LABELS_TO_NAMES = {0: 'drone'}
ROOTDIR = "/media/alejandro/DATA/datasets/simple_m100_dataset_test"

if SAVE_ANNOTATIONS:
    # Store prediction information
    with open(ROOTDIR + '/predictions.txt', 'a') as the_file:
        the_file.write('score,correct,iou,gt_class\n')

with open(ROOTDIR + '/manual_annotations.txt', 'rb') as csvfile:
    groundtruth = csv.DictReader(csvfile, delimiter=',', quotechar='|')

    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    # model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
    # model_path = os.path.join('..', 'snapshots', 'resnet50_csv_02_m100_inference.h5')
    model_path = "/media/alejandro/DATA/temp/unreal_resnet50_csv_14.h5"

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    model = models.convert_model(model)

    #print(model.summary())
    # Go across all files and subfolders
    for row in groundtruth:
        # load image
        image = read_image_bgr(row['path'])

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.7:
                break

            b = box.astype(int)
            b_gt = [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])]

            # Compute IoU
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(b[0], b_gt[0])
            yA = max(b[1], b_gt[1])
            xB = min(b[2], b_gt[2])
            yB = min(b[3], b_gt[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
            boxBArea = (b_gt[2] - b_gt[0] + 1) * (b_gt[3] - b_gt[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)

            if SAVE_ANNOTATIONS:
                if LABELS_TO_NAMES[label] == row['class']:
                    print("Correct with IoU " + str(iou) + " and score " + str(score))
                    # Store prediction information
                    with open(ROOTDIR + '/predictions.txt', 'a') as the_file:
                        the_file.write(str(score) + ',' + '1' + ',' + str(iou) + ',' + str(LABELS_TO_NAMES.values().index(row['class'])) + '\n')
                else:
                    print("Not correct with IoU " + str(iou) + " and score " + str(score))
                    # Store prediction information
                    with open(ROOTDIR + '/predictions.txt', 'a') as the_file:
                        the_file.write(str(score) + ',' + '0' + ',' + str(iou) + ',' + str(LABELS_TO_NAMES.values().index(row['class'])) + '\n')

            if SHOW_IMAGES:
                color = label_color(label)
                color_gr = (0, 255, 255)

                draw_box(draw, b, color=color)
                draw_box(draw, b_gt, color=color_gr)

                caption = "{} {:.3f}".format(LABELS_TO_NAMES[label], score)
                print("Score: " + str(score))
                draw_caption(draw, b, caption)

        if SHOW_IMAGES:
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(draw)
            plt.show()