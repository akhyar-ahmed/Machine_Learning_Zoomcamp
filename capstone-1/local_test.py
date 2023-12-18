"""

To run it:
python local_test.py

"""
import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Open the JSON file
with open('class_labels.json', 'r') as f:
    # Load the JSON data
    data = json.load(f)

# Now 'classes' are in a dictionary
classes = list(data.keys())


# Load the best model
best_model = load_model('models/best_model_notebook_1.h5')
img_path = 'small_dataset/images_for_test/TomatoYellowCurlVirus1.JPG'  # Provide the path to your image
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Get the predictions
predictions = best_model.predict(img_array)

# Assuming it's a multiclass classification, you can get the class with the highest probability
predicted_class = np.argmax(predictions)

# Print the result
print(f'Predicted class: {predicted_class}, {classes[predicted_class]}')
