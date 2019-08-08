import argparse
import json

import numpy as np
import requests
import cv2

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
data = cv2.imread(image_path)
data = cv2.resize(data, (416, 416))
data = data / 255.
data = data[:, :, ::-1]
data = data.astype(np.float32)
data = np.expand_dims(data, 0)

payload = {
    "inputs": [{'input': data.tolist()}],
    "signature_name": "predict_images" # Custom signature name
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/ObjectDetector:predict', json=payload)
predictions = json.loads(r.content.decode('utf-8'))

print(predictions)
