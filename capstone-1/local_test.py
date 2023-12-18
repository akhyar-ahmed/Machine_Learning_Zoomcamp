"""

To run it:
python local_test.py

"""
import json
import numpy as np
import tensorflow.lite as tflite

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Open the JSON file
with open('class_labels.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

# Now 'classes' are in a dictionary
classes = list(data.keys())

img_path = 'images_for_test/TomatoYellowCurlVirus1.JPG'  # Provide the path to your image

############# If you use tflite model then uncomment this part of the code and comment below part of the code #################

# Load the best model
model_path = 'models/best_model.tflite'
best_model = tflite.Interpreter(model_path=model_path)

best_model.allocate_tensors()
input_index = best_model.get_input_details()[0]['index']
output_index = best_model.get_output_details()[0]['index']

img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# invoke inputs
best_model.set_tensor(input_index, img_array)
best_model.invoke()
# Get the predictions
predictions = best_model.get_tensor(output_index)
################################################################################################################################


# ############# If you use .h5 model then uncomment this part of the code and comment above part of the code #####################

# Load the best model
# model_path = 'models/best_model.h5'

# best_model = load_model(model_path)
# img = image.load_img(img_path, target_size=(256, 256))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# # Get the predictions
# predictions = best_model.predict(img_array)
# ################################################################################################################################


# Assuming it's a multiclass classification, you can get the class with the highest probability
predicted_class = np.argmax(predictions)

# Print the result
print(f'\nPredicted class: {predicted_class}, {classes[predicted_class]}')
