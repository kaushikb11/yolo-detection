from darkflow.net.build import TFNet
import cv2
import argparse
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

options = {"model": "./cfg/tiny-yolo-voc-fynd.cfg",
           "load": 875,
           "threshold": 0.30}

tfnet = TFNet(options)

image_path = args['image']
image = cv2.imread(image_path)
result = tfnet.return_predict(image)

for res in result:
    print("{} (confidence: {})".format(res['label'], str(res['confidence'])))
    print("topleft({}, {}), botright({}, {})".format(res['topleft']['x'], res['topleft']['y'],
                                                    res['bottomright']['x'], res['bottomright']['y']))
