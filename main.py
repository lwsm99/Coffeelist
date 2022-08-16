# Main Script which initializes the app
from detect_objects import *

# Initialize app
if __name__ == '__main__':
    img = 'Tensorflow/workspace/images/collectedimages/tables/CL_DATA_060822 (1).jpeg'

    names = detect_names(img)
    objects = detect_strokes(img)
