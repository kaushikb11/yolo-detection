from darkflow.net.build import TFNet

def get_model():
    options = {"model": "../config/tiny-yolo-voc-fynd.cfg",
        "load": 875,
        "threshold": 0.30}

    tfnet = TFNet(options)
    return tfnet


