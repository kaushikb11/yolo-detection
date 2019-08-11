import argparse
import json

import numpy as np
import requests
import cv2
import sys

sys.path.append('..')
from utils import get_boxes_with_labels, preprocess_image_data

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

SERVER_URL = 'http://localhost:9000/v1/models/darkflow:predict'

image_path = args['image']
data = preprocess_image_data(image_path)

payload = {
    "signature_name": "predict",  # Custom signature name
    "instances": [{'input': data.tolist()}]
}

# sending post request to TensorFlow Serving server
response = requests.post(SERVER_URL, json=payload)

json_response = json.loads(response.text)
net_out = np.squeeze(np.array(json_response['predictions'], dtype='float32'))

predictions = get_boxes_with_labels(net_out, data)
print(predictions)
