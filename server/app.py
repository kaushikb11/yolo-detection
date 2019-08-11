import base64
import json
from io import BytesIO
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.preprocessing import image
import cv2
import sys

sys.path.append('..')
from utils import get_boxes_with_labels

app = Flask(__name__)


@app.route('/darkflow/predict/', methods=['POST'])
def image_classifier():
    SERVER_URL = 'http://localhost:9000/v1/models/darkflow:predict'
    # Convert string of image data to uint8
    nparr = np.fromstring(request.data, np.uint8)
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    data = cv2.resize(img, (416, 416))
    data = data / 255.
    data = data[:, :, ::-1]
    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input': data.tolist()}],
        "signature_name": "predict"  # Custom signature name
    }

    # Making POST request
    response = requests.post(SERVER_URL, json=payload)
    json_response = json.loads(response.text)
    net_out = np.squeeze(np.array(json_response['predictions'], dtype='float32'))
    
    predictions = get_boxes_with_labels(net_out, data)
    return jsonify({'results': str(predictions)})


if __name__ == "__main__":
    app.run(debug=True, port=4000)
