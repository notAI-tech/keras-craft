import logging
import data_utils
from craft_model import build_craft_vgg_model


class Detector():
    model = build_craft_vgg_model()
    def __init__(self, weights_path):
        Detector.model.load_weights(weights_path)

    def detect(self, image_paths, max_width=720, max_height=None):
        if not (max_width or max_height):
            logging.warn("Both max_width and max_height are set to None. Only do this if you are aware of the implications.")

        loaded_images, scales_before_padding, shapes_before_padding = data_utils.read_images(image_paths, max_width = max_width, max_height = max_height)

        model_predictions = Detector.model.predict(loaded_images)

        boxes = data_utils.keras_pred_to_boxes(model_predictions, shapes=shapes_before_padding, scales=scales_before_padding)

        return boxes
