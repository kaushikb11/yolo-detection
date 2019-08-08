from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import numpy as np
import sys
import cv2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', sys.argv[1], sys.argv[1])
FLAGS = tf.app.flags.FLAGS


def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    data = cv2.imread(sys.argv[1])
    data = cv2.resize(data, (416, 416))
    data = data.astype(np.float32)
    data = tf.contrib.util.make_tensor_proto(data, shape=[1, 416, 416, 3])

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'object_detector'  # Export directory of SavedModel
    request.model_spec.signature_name = "predict_images"

    request.inputs['input'].CopyFrom(data)
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    print(result)


if __name__ == '__main__':
    tf.app.run()
