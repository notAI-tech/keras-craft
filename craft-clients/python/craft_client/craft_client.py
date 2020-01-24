import data_utils

import numpy as np
import grpc
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel("127.0.0.1:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = "craft-text-detection"
request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

def predict_response_to_array(response, output_tensor_name):
    dims = response.outputs[output_tensor_name].tensor_shape.dim
    shape = tuple(d.size for d in dims)
    return np.reshape(response.outputs[output_tensor_name].float_val, shape)

def detect(image_paths, max_width=720, max_height=None):
    if not (max_width or max_height):
        logging.warn("Both max_width and max_height are set to None. Only do this if you are aware of the implications.")

    loaded_images, scales_before_padding, shapes_before_padding = data_utils.read_images(image_paths, max_width = max_width, max_height = max_height)

    request.inputs["input"].CopyFrom(tf.contrib.util.make_tensor_proto(loaded_images, shape=None))

    response = stub.Predict(request, 60)

    model_predictions = predict_response_to_array(response, "prediction")


    boxes = data_utils.keras_pred_to_boxes(model_predictions, shapes=shapes_before_padding, scales=scales_before_padding)

    return boxes
