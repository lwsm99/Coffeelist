# Object detection script which detects names, counts strokes and labels them (1 or 5), assigns the bounding boxes to names and then counts them individually

### IMPORTS ###
import os
import cv2
import pytesseract
import numpy as np
import re


### GLOBAL VARIABLES ###

# Name Detection
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
global names

# Stroke Detection
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


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

    global names
    names = [['0' for _ in range(5)] for _ in range(20)]
    last_el = 0

    # Get average x position for data with a confidence rate of > 66%
    for x, d in enumerate(data.splitlines()):
        if x != 0:
            d = d.split()
            # Only work with data where text has been detected and confidence rate > 66%
            if len(d) == 12 and float(d[10]) > 66:
                avg_count += 1
                avg_x_start += int(d[6])
                avg_x_end += int(d[6]) + int(d[8])
    avg_x_start = (avg_x_start / avg_count) - 250
    avg_x_end = (avg_x_end / avg_count) + 250

    # Detect Names by looping through data
    for x, d in enumerate(data.splitlines()):
        if x != 0:
            d = d.split()
            # Only work with data where text has been detected and text is inbetween average box
            if len(d) == 12 and int(d[6]) > avg_x_start and int(d[6]) + int(d[8]) < avg_x_end:
                x, y, w, h = int(d[6]), int(d[7]), int(d[8]), int(d[9])
                # Create Bounding Box for Names
                cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)

                if last_el == 0:
                    # Name, x, y, width, height
                    names[0][0], names[0][1], names[0][2], names[0][3], names[0][4] = d[11], d[6], d[7], d[8], d[9]
                    last_el += 1
                else:
                    # If box is around the same height, append to previous saved box (First + Last Name)
                    if int(names[last_el - 1][2]) - 25 < int(d[7]) < int(names[last_el - 1][2]) + 25:
                        names[last_el - 1][0] += d[11]   # Name
                        names[last_el - 1][2] = str(max(int(names[last_el - 1][2]), int(d[7])))   # y
                        names[last_el - 1][3] = str((int(d[6]) + int(d[8])) - int(names[last_el - 1][1]))   # width
                        names[last_el - 1][4] = str(max(int(names[last_el - 1][4]), int(d[9])))  # height
                    else:
                        names[last_el][0], names[last_el][1], names[last_el][2], names[last_el][3], names[last_el][4] = d[11], d[6], d[7], d[8], d[9]
                        last_el += 1

    for x, n in enumerate(names):
        # Add Spaces inbetween names
        n[0] = re.sub(r"(\w)([A-Z])", r"\1 \2", n[0])
        # Delete empty list elements
        if n[0] == '0':
            del names[x:len(names)]

    print(np.matrix(names))

    # TODO: Cleanup random inputs that aren't names (When x is >= a detected stroke)

    # Show Image for testing purposes
    img = cv2.resize(img, (500, 666))
    cv2.imshow('Result', img)
    cv2.waitKey(0)


# Detect single strokes & blocks of 5 and assign labels
def detect_strokes():
    print("Counting Strokes...")


# Assign bounding boxes to detected name individually using coordinates
def get_count_for_name():
    print("Counting for each person...")
