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
# LABELS_TO_NAMES = {0: 'drone', 1: 'bird', 2: 'car'}
SHOW_IMAGES = 1
SAVE_ANNOTATIONS = 1
LABELS_TO_NAMES = {0: 'drone', 1: 'bird', 2: 'car'}
ROOTDIR = "/media/alejandro/DATA/datasets/real_uav_raw_dataset"
OUTDIR = "/media/alejandro/DATA/datasets/real_uav_dataset"

if SAVE_ANNOTATIONS:
    # Create subdirectories
    os.mkdir(OUTDIR + '/detected', 0755)
    os.mkdir(OUTDIR + '/not_detected', 0755)
    os.mkdir(OUTDIR + '/data', 0755)
    os.mkdir(OUTDIR + '/data/obj', 0755)

    # Store names information
    with open(OUTDIR + '/data/obj.names', 'a') as the_file:
        the_file.write(LABELS_TO_NAMES[0])

    # Store data information
    with open(OUTDIR + '/data/obj.data', 'a') as the_file:
        the_file.write('classes = 1\n')
        the_file.write('train  = data/train.txt\n')
        the_file.write('valid  = data/test.txt\n')
        the_file.write('names = data/obj.names\n')
        the_file.write('backup = backup/\n')

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
# model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
# model_path = os.path.join('..', 'snapshots', 'resnet50_csv_02_m100_inference.h5')
model_path = "/media/alejandro/DATA/Shared/paper_experiments/paper-uav-following-iros-2019/experiments/data/resnet50_csv_04_m100/resnet50_csv_04.h5"

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# Counter
counter = 0

#print(model.summary())
# Go across all files and subfolders
for subdir, dirs, files in os.walk(ROOTDIR):
    for file in sorted(files):
        print("Raw read: " + os.path.join(subdir, file))
        if file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".png"):
            # Disable flag
            bb_found = False

            try:
                # load image
                image = read_image_bgr(os.path.join(subdir, file))
            except:
                continue

            # copy to draw on
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            img = preprocess_image(image)
            img, scale = resize_image(img)

            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
            print("processing time: ", time.time() - start)
            print("with scale: " + str(scale))

            # Save witdth and height
            height, width, channels = draw.shape
            print(draw.shape)

            # correct for image scale
            boxes /= scale

            # visualize detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.6 or label != 0:
                    break

                # Enable flag
                bb_found = True

                b = box.astype(float)
                print(b)
                x1 = b[0]
                y1 = b[1]
                x2 = b[2]
                y2 = b[3]

                bbwidth = (x2 - x1)
                bbheight = (y2 - y1)
                cx = (x1 + (bbwidth / 2.0)) / float(width)
                cy = (y1 + (bbheight / 2.0)) / float(height)
                bbwidth = (x2 - x1) / float(width)
                bbheight = (y2 - y1) / float(height)


                if SAVE_ANNOTATIONS:
                    # Remove extension
                    outfile = file.replace('.jpg', '')
                    outfile = outfile.replace('.jpeg', '')
                    outfile = outfile.replace('.JPG', '')
                    outfile = outfile.replace('.JPEG', '')
                    outfile = outfile.replace('.png', '')
                    outfile = outfile.replace('.GIF', '')
                    outfile = outfile.replace('.gif', '')

                    # Store names information
                    with open(os.path.join(OUTDIR + '/data/obj', str(counter).zfill(6) + '_' + outfile + '.txt'), 'a') as the_file:
                        the_file.write(str(label) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(bbwidth) + ' ' + str(bbheight) + '\n')

                #if SHOW_IMAGES:
                    color = label_color(label)
                    color_gr = (0, 255, 255)

                    draw_box(draw, b, color=color)

                    caption = "{} {:.3f}".format(LABELS_TO_NAMES[label], score)
                    print("Score: " + str(score))
                    draw_caption(draw, b, caption)


            if bb_found:
                # Save image in dataset
                cv2.imwrite(os.path.join(OUTDIR + '/data/obj', str(counter).zfill(6) + '_' + file), image)
                # Save examples
                draw_bgr = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(OUTDIR + '/detected', str(counter).zfill(6) + '_' + file), draw_bgr)
                # Store names information
                with open(os.path.join(OUTDIR + '/data', 'train.txt'), 'a') as the_file:
                    the_file.write(os.path.join('data/obj', str(counter).zfill(6) + '_' + file) + '\n')
            else:
                # Save image in dataset
                cv2.imwrite(os.path.join(OUTDIR + '/not_detected', str(counter).zfill(6) + '_' + file), image)

            # Increase counter
            counter += 1

            if SHOW_IMAGES:
                plt.figure(figsize=(15, 15))
                plt.axis('off')
                plt.imshow(draw)
                plt.show()
