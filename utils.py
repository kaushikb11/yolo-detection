from model import get_model


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
