""""
Convert a model from Keras to TF-Lite format.

To Run it:
python create_light_model.py
"""


import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = "./models/best_model.h5"

# Load the saved Keras model
loaded_model = load_model(model_path)

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('./models/best_model.tflite', 'wb') as f:
    f.write(tflite_model)


# Print the size of the converted model
converted_model_size = len(tflite_model) / (1024 * 1024)  # Size in megabytes
print(f"Size of the converted model: {converted_model_size:.2f} MB")