import os
import pydload
import logging
from . import data_utils
from .craft_model import build_craft_vgg_model

model_links = {
                'generic-english': 'https://github.com/bedapudi6788/keras-craft/releases/download/checkpoint-release/generic-english'
            }

model_alternate_names = {
    'default': 'generic-english'
}

class Detector():
    model = build_craft_vgg_model()
    def __init__(self, model_name = 'default'):
        if model_name in model_alternate_names:
            model_name = model_alternate_names[model_name]

        home = os.path.expanduser("~")
        checkpoint_dir = os.path.join(home, '.keras_craft_' + model_name)
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint')

        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        
        if not os.path.exists(checkpoint_path):
            print('Downloading checkpoint', model_links[model_name], 'to', checkpoint_path)
            pydload.dload(url=model_links[model_name], save_to_path=checkpoint_path, max_time=None)
        
        Detector.model.load_weights(checkpoint_path)

    def detect(self, image_paths, max_width=720, max_height=None):
        '''
        Function for reading image(s), running predictions.

        :param image_paths: single or list of opencv images or image file paths
        :max_width: images are scaled to this width keeping the aspect ratio.
        :max_height: either max_width or max_height should be used.

        :return: list of numpy arrays. each array is all the detected boxes in the corresponding image.
        '''
        return_single = False

        if not isinstance(image_paths, list):
            logging.warn('Batch for better performance')
            return_single = True
            image_paths = [image_paths]

        if not (max_width or max_height):
            logging.warn("Both max_width and max_height are set to None. Only do this if you are aware of the implications.")

        loaded_images, scales_before_padding, shapes_before_padding = data_utils.read_images(image_paths, max_width = max_width, max_height = max_height)

        model_predictions = Detector.model.predict(loaded_images)

        boxes = data_utils.keras_pred_to_boxes(model_predictions, shapes=shapes_before_padding, scales=scales_before_padding)

        if return_single:
            boxes = boxes[0]

        return boxes
