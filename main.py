# Main Script which initializes the app
from detect_objects import *
from export_data import *
import numpy as np

# Initialize app
if __name__ == '__main__':
    img = 'Tensorflow/workspace/images/collectedimages/tables/CL_DATA_060822 (2).jpeg'

    objects = detect_strokes(img)
    names = detect_names(img)
    names = get_count_for_name(objects, names)
    export_data(names)

    print(np.matrix(names))
