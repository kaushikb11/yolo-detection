from model import get_model
import cv2


def get_boxes_with_labels(array, data):
    predictions = []
    tfnet = get_model()
    boxes = tfnet.framework.findboxes(array)
    h, w, _ = data.shape
    threshold = tfnet.FLAGS.threshold
    for box in boxes:
        tmpBox = tfnet.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        predictions.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })

    return predictions


def preprocess_image_data(image_path):
    data = cv2.imread(image_path)
    data = cv2.resize(data, (416, 416))
    data = data / 255.
    data = data[:, :, ::-1]
    return data
