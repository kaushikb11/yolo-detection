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
image = cv2.resize(image, (416, 416))
result = tfnet.return_predict(image)


for res in result:
    x1, y1 = res['topleft']['x'], res['topleft']['y']
    x2, y2 = res['bottomright']['x'], res['bottomright']['y']
    print("{} (confidence: {})".format(res['label'], str(res['confidence'])))
    print("topleft({}, {}), botright({}, {})".format(x1, y1, x2, y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(image, res['label'] + ':' + str(res['confidence']), (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.imwrite("../demo_images/{}.jpg".format(res['label']), image)
    cv2.imshow("test", image)
    cv2.waitKey(0)