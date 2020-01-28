# keras-craft
Extremely easy to use Text Detection module with CRAFT pre-trained model.

keras-craft aims to be production ready and supports features like batch inference (auto batching for images of different size) and tensorflow serving.

# Installation

```pip install --upgrade craft_client``` (uses tf-serving - **recommended**)

```pip install keras_craft``` (the entire library)

# Usage (craft_client)
```bash
docker run -p 8500:8500 bedapudi6788/keras-craft:generic-english
```
```python
import craft_client

image_paths = [image_1, image_2, ..]
all_boxes = craft_client.detect(image_paths)

# Visualization
for image_path, boxes in zip(image_paths):
  image_with_boxes_path = craft_client.draw_boxes_on_image(image_path, boxes)
  print(image_with_boxes_path)
```

# Usage (keras_craft)
```python
import keras_craft

detector = keras_craft.Detector()

image_paths = [image_1, image_2, ..]
all_boxes = detector.detect(image_paths)

# Visualization
for image_path, boxes in zip(image_paths):
  image_with_boxes_path = keras_craft.draw_boxes_on_image(image_path, boxes)
  print(image_with_boxes_path)
```

# To Do:

1. Train different models for different use-cases. (various languages ..)
2. Experiment with smaller model(s)



**Credit for the core keras model, generic-english checkpoint .. goes to [Fausto Morales](https://github.com/faustomorales/keras-ocr) and Clova.ai**
