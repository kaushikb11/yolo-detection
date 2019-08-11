import argparse
import cv2
import requests
import json
import ast


# defining the api-endpoint
API_ENDPOINT = "http://localhost:4000/darkflow/predict/"

# taking input image via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread(image_path)
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)

# sending post request and saving response as response object
response = requests.post(url=API_ENDPOINT, data=img_encoded.tostring(), headers=headers)

# extracting the response
json_response = json.loads(response.text)
predictions = ast.literal_eval(json_response['results'])
for pred in predictions:
    print(pred)
