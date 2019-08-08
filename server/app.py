import base64
import json
from io import BytesIO
import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.preprocessing import image

app = Flask(__name__)


@app.route('/objectdetector/detect/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
                                            target_size=(416, 416))) / 255.
    img = np.expand_dims(img, axis=0)
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "inputs": [{'input': img.tolist()}],
        "signature_name": "predict_images"  # Custom signature name
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/ObjectDetector:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    predictions = json.loads(r.content.decode('utf-8'))

    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=True, port=4000)
