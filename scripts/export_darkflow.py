from darkflow.net.build import TFNet
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder

options = {"model": "./cfg/tiny-yolo-voc-fynd.cfg",
           "load": 875,
           "threshold": 0.30}

tfnet = TFNet(options)

image = cv2.imread("../test_images/backstrap9.jpg")
result = tfnet.return_predict(image)
for res in result:
    print("{} (confidence: {})".format(res['label'], str(res['confidence'])))
    print("topleft( {}, {}), botright({}, {})".format(res['topleft']['x'], res['topleft']['y'],
                                                    res['bottomright']['x'], res['bottomright']['y']))

export_path = '../object_detector/1'
print('Exporting trained model to', export_path)

builder = saved_model_builder.SavedModelBuilder(export_path)

tensor_info_x = tf.saved_model.utils.build_tensor_info(tfnet.inp)
tensor_info_y = tf.saved_model.utils.build_tensor_info(tfnet.out)

prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    inputs={'input': tensor_info_x},
    outputs={'output': tensor_info_y},
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder.add_meta_graph_and_variables(
    tfnet.sess, [tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        'predict_images':
            prediction_signature,
    },
    legacy_init_op=legacy_init_op)

builder.save()
print('Done exporting!')
