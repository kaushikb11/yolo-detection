from darkflow.net.build import TFNet
import cv2
import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import sys

sys.path.append('..')
from model import get_model


tfnet = get_model()
image_path = "../test_images/backstrap9.jpg"

image = cv2.imread(image_path)
result = tfnet.return_predict(image)
for res in result:
    print("{} (confidence: {})".format(res['label'], str(res['confidence'])))
    print("topleft({}, {}), botright({}, {})".format(res['topleft']['x'], res['topleft']['y'],
                                                    res['bottomright']['x'], res['bottomright']['y']))


number_of_versions = len(os.listdir('../darkflow/'))
export_path = '../darkflow/{}'.format(str(number_of_versions + 1))
print('Exporting trained model to', export_path)

with tfnet.sess.graph.as_default():
    x_op = tfnet.sess.graph.get_operation_by_name("input")
    x = x_op.outputs[0]
    pred_op = tfnet.sess.graph.get_operation_by_name("output")
    pred = pred_op.outputs[0]

with tfnet.sess.graph.as_default():
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            "input": tf.saved_model.utils.build_tensor_info(x)
        },
        outputs={
            "output": tf.saved_model.utils.build_tensor_info(pred)
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        tfnet.sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict": prediction_signature,
        })
        
    builder.save()
    print('Done exporting!')
