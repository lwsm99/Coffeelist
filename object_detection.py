# Object detection script which detects names, counts strokes and labels them (1 or 5), assigns the bounding boxes to names and then counts them individually

### IMPORTS ###
import os
import cv2
import pytesseract


### GLOBAL VARIABLES ###

# Name Detection
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

    # Detect Names by looping through data
    for x, d in enumerate(data.splitlines()):
        if x != 0:
            d = d.split()
            # Only work with data where text has been detected
            if len(d) == 12:
                x, y, w, h = int(d[6]), int(d[7]), int(d[8]), int(d[9])
                # Create Bounding Box for Names
                cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 3)
                print(d[11])
                # cv2.putText(img, d[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

    # TODO: Save First & Last Names as one (When y is similar, combine)
    # TODO: Cleanup random inputs that aren't names (When x is too far apart + when x is >= a detected stroke)
    # TODO: Save cleaned up Names into a List

    # Show Image for testing purposes
    cv2.imshow('Result', img)
    cv2.waitKey(0)


# Detect single strokes & blocks of 5 and assign labels
def detect_strokes():
    print("Counting Strokes...")


# Assign bounding boxes to detected name individually using coordinates
def get_count_for_name():
    print("Counting for each person...")
