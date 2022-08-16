# Object detection script which detects names, counts strokes and labels them (1 or 5), assigns the bounding boxes to names and then counts them individually

### IMPORTS ###
import os
import cv2
import pytesseract
import numpy as np
import re
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import matplotlib
from matplotlib import pyplot as plt
import tkinter
matplotlib.use('TkAgg')


### GLOBAL VARIABLES ###

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

configs = config_util.get_configs_from_pipeline_file('Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('Tensorflow/workspace/models/my_ssd_mobnet/ckpt-7').expect_partial()
category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')


### METHODS ###

# Detect names
def detect_names(image: str):

    # Variables
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cfg = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "'
    data = pytesseract.image_to_data(img, config=cfg)

    avg_count = 0
    avg_x_start = 0
    avg_x_end = 0
    avg_buffer = 250

    names = [['0' for _ in range(5)] for _ in range(50)]
    last_el = 0

    # Calculate average x position for data with a confidence rate of > 66%
    for x, d in enumerate(data.splitlines()):
        if x != 0:
            d = d.split()
            # Only work with data where text has been detected and confidence rate > 66%
            if len(d) == 12 and float(d[10]) > 66:
                avg_count += 1
                avg_x_start += int(d[6])
                avg_x_end += int(d[6]) + int(d[8])
    avg_x_start = (avg_x_start / avg_count) - avg_buffer
    avg_x_end = (avg_x_end / avg_count) + avg_buffer

    # Detect names by looping through data
    for x, d in enumerate(data.splitlines()):
        if x != 0:
            d = d.split()
            # Only work with data where text has been detected and text is inside average limit
            if len(d) == 12 and int(d[6]) > avg_x_start and int(d[6]) + int(d[8]) < avg_x_end:
                x, y, w, h = int(d[6]), int(d[7]), int(d[8]), int(d[9])
                # Create Bounding Box for Names
                cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)

                if last_el == 0:
                    # Name, x, y, width, height
                    names[0][0], names[0][1], names[0][2], names[0][3], names[0][4] = d[11], d[6], d[7], d[8], d[9]
                    last_el += 1
                else:
                    # If box is around the same height, append to previous saved box (first + last name)
                    if int(names[last_el - 1][2]) - 25 < int(d[7]) < int(names[last_el - 1][2]) + 25:
                        names[last_el - 1][0] += d[11]   # Name
                        names[last_el - 1][2] = str(max(int(names[last_el - 1][2]), int(d[7])))   # Y
                        names[last_el - 1][3] = str((int(d[6]) + int(d[8])) - int(names[last_el - 1][1]))   # Width
                        names[last_el - 1][4] = str(max(int(names[last_el - 1][4]), int(d[9])))  # Height
                    else:
                        names[last_el][0], names[last_el][1], names[last_el][2], names[last_el][3], names[last_el][4] = d[11], d[6], d[7], d[8], d[9]
                        last_el += 1

    for x, n in enumerate(names):
        # Add Spaces inbetween names
        n[0] = re.sub(r"(\w)([A-Z])", r"\1 \2", n[0])
        # Delete empty list elements
        if n[0] == '0':
            del names[x:len(names)]

    # TODO: Cleanup random inputs that aren't names (When x is >= a detected stroke)

    # DEBUG
    # img = cv2.resize(img, (500, 666))
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)

    # print(np.matrix(names))

    return names


# Detect single strokes & blocks of 5 and assign labels
def detect_strokes(image: str):
    img = cv2.imread(image)
    image_np = np.array(img)
    img_h, img_w, _ = img.shape

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1000,
        min_score_thresh=.3,
        agnostic_mode=False)

    objects = [['0' for _ in range(5)] for _ in range(len(detections['detection_classes']))]

    for i in range(len(detections['detection_classes'])):
        # Set name for every object
        if detections['detection_classes'][i] == 0:
            objects[i][0] = "Block"
        else:
            objects[i][0] = "Single"

        # Set x, y, w & h for every object
        objects[i][1] = str(round(detections['detection_boxes'][i][1] * img_w))   # x
        objects[i][2] = str(round(detections['detection_boxes'][i][0] * img_h))   # y
        objects[i][3] = str(round(detections['detection_boxes'][i][3] * img_w - int(objects[i][1])))  # w
        objects[i][4] = str(round(detections['detection_boxes'][i][2] * img_h - int(objects[i][2])))  # h

    # DEBUG
    image_np_with_detections = cv2.resize(image_np_with_detections, (750, 1000))
    cv2.imshow('Result', image_np_with_detections)
    cv2.waitKey(0)
    # plt.imshow(image_np_with_detections)
    # plt.show()
    # print(np.matrix(objects))

    return objects


# Assign bounding boxes to detected name individually using coordinates
def get_count_for_name():
    print("Counting for each person...")


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
